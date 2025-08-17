import os
import json
import tempfile
import subprocess
import asyncio
import base64
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import re
from dataclasses import dataclass
from contextlib import asynccontextmanager
import aiohttp

import google.generativeai as genai
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
import uvicorn


@dataclass
class QuestionResult:
    """Structured result for each question"""
    question_number: int
    question_text: str
    result: Any
    success: bool
    error_message: Optional[str] = None
    execution_time: float = 0.0


class URLExtractor:
    """Extract URLs from text content"""
    
    @staticmethod
    def extract_urls(text: str) -> List[str]:
        """Extract all URLs from text using regex"""
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+[^\s<>"{}|\\^`\[\].,;:!?]'
        urls = re.findall(url_pattern, text, re.IGNORECASE)
        return list(set(urls))  # Remove duplicates


class ScraperClient:
    """Simplified scraper client using only 8000 route"""
    
    @staticmethod
    async def wake_up_scraper() -> None:
        """Send minimal request to wake up the scraper service"""
        try:
            async with aiohttp.ClientSession() as session:
                data = aiohttp.FormData()
                data.add_field('questions_txt', 'wake up', filename='questions.txt', content_type='text/plain')
                
                async with session.post("https://scrp-vrz3.onrender.com/api/", data=data, timeout=10) as response:
                    print(f"🔄 Scraper wake-up: {response.status}")
        except Exception as e:
            print(f"🔄 Scraper wake-up failed (non-critical): {e}")
    
    @staticmethod
    async def scrape_with_questions(questions_content: str) -> Dict[str, Any]:
        """Call scraper API with questions.txt content and return structured response"""
        async with aiohttp.ClientSession() as session:
            # Create multipart form data
            data = aiohttp.FormData()
            data.add_field('questions_txt', questions_content, 
                          filename='questions.txt', content_type='text/plain')
            
            try:
                async with session.post("https://scrp-vrz3.onrender.com/api/", data=data, timeout=300) as response:
                    if response.status == 200:
                        result = await response.json()
                        print(f"✅ Scraper returned: {len(str(result))} chars")
                        return result  # This is the structured response like your curl
                    else:
                        error_text = await response.text()
                        raise Exception(f"Scraper API failed with status {response.status}: {error_text}")
            except asyncio.TimeoutError:
                raise Exception("Scraper API timeout (300s)")
            except Exception as e:
                raise Exception(f"Scraper API error: {str(e)}")


class FilePreviewGenerator:
    """Generate comprehensive previews for different file formats"""
    
    @staticmethod
    def preview_json(filepath: Path) -> Dict[str, Any]:
        """Preview JSON file structure"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, dict):
                keys = list(data.keys())[:20]  # First 20 keys
                sample_values = {}
                for key in keys[:5]:  # Sample first 5 key-value pairs
                    val = data[key]
                    if isinstance(val, (list, dict)):
                        sample_values[key] = f"{type(val).__name__} with {len(val)} items"
                    else:
                        sample_values[key] = str(val)[:100]
                
                return {
                    "type": "json_object",
                    "keys": keys,
                    "total_keys": len(data.keys()),
                    "sample_data": sample_values,
                    "structure": "nested object" if any(isinstance(v, (dict, list)) for v in data.values()) else "flat object"
                }
            
            elif isinstance(data, list):
                return {
                    "type": "json_array", 
                    "length": len(data),
                    "sample_items": data[:3] if len(data) > 0 else [],
                    "item_types": [type(item).__name__ for item in data[:10]]
                }
                
        except json.JSONDecodeError:
            # Try as JSONL
            try:
                lines = []
                with open(filepath, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        if i >= 3:  # Only sample first 3 lines
                            break
                        lines.append(json.loads(line.strip()))
                
                return {
                    "type": "jsonl",
                    "sample_lines": lines,
                    "format": "JSON Lines (JSONL)"
                }
            except:
                return {"error": "Invalid JSON format"}
        except Exception as e:
            return {"error": str(e)}

    @staticmethod 
    def preview_parquet(filepath: Path) -> Dict[str, Any]:
        """Preview Parquet file structure"""
        try:
            import pandas as pd
            import pyarrow.parquet as pq
            
            # Get basic info without loading full file
            parquet_file = pq.ParquetFile(filepath)
            schema = parquet_file.schema
            
            # Load small sample
            df_sample = pd.read_parquet(filepath, nrows=1000)  # First 1000 rows
            
            return {
                "type": "parquet",
                "rows": parquet_file.metadata.num_rows,
                "columns": len(schema),
                "column_info": {
                    col.name: str(col.physical_type) for col in schema
                },
                "sample_columns": list(df_sample.columns),
                "sample_dtypes": df_sample.dtypes.to_dict(),
                "memory_usage": f"{df_sample.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB (sample)",
                "sample_data": df_sample.head(3).to_dict('records') if len(df_sample) > 0 else []
            }
            
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def preview_image(filepath: Path) -> Dict[str, Any]:
        """Preview image file"""
        try:
            from PIL import Image
            import pytesseract
            
            with Image.open(filepath) as img:
                # Basic image info
                info = {
                    "type": "image",
                    "format": img.format,
                    "mode": img.mode,
                    "size": img.size,
                    "megapixels": round((img.size[0] * img.size[1]) / 1_000_000, 2)
                }
                
                # Try OCR for text extraction (for tables in images)
                try:
                    # Convert to RGB if needed
                    if img.mode != 'RGB':
                        img_rgb = img.convert('RGB')
                    else:
                        img_rgb = img
                    
                    # Quick OCR sample
                    text_sample = pytesseract.image_to_string(img_rgb)[:500]
                    
                    if text_sample.strip():
                        info["has_text"] = True
                        info["text_sample"] = text_sample.strip()
                        
                        # Check if it might contain tabular data
                        lines = text_sample.split('\n')
                        tabular_indicators = sum(1 for line in lines if len(line.split()) > 3)
                        info["likely_table"] = tabular_indicators > 2
                    else:
                        info["has_text"] = False
                        
                except Exception:
                    info["ocr_available"] = False
                    
                return info
                
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def preview_pdf(filepath: Path) -> Dict[str, Any]:
        """Enhanced PDF preview with robust table detection"""
        try:
            import pdfplumber
            import pandas as pd
            
            with pdfplumber.open(filepath) as pdf:
                page_count = len(pdf.pages)
                
                if page_count == 0:
                    return {'error': 'Empty PDF'}
                
                # Sample multiple pages for better coverage
                sample_pages = min(5, page_count)  
                all_text = ""
                all_tables = []
                table_previews = []
                
                for i in range(sample_pages):
                    page = pdf.pages[i]
                    
                    # Extract text
                    page_text = page.extract_text() or ""
                    all_text += page_text + "\n"
                    
                    # Try multiple table extraction strategies
                    strategies = [
                        {"vertical_strategy": "lines", "horizontal_strategy": "lines"},
                        {"vertical_strategy": "text", "horizontal_strategy": "text"},
                        {"edge_min_length": 3, "snap_tolerance": 3},
                        {"vertical_strategy": "lines", "horizontal_strategy": "text"},
                        {}  # Default settings
                    ]
                    
                    best_tables = []
                    for strategy in strategies:
                        try:
                            page_tables = page.extract_tables(table_settings=strategy)
                            if page_tables and len(page_tables) > 0:
                                # Quality check - prefer tables with more consistent column counts
                                quality_score = sum(len(row) for table in page_tables for row in table if row)
                                if quality_score > sum(len(row) for table in best_tables for row in table if row):
                                    best_tables = page_tables
                        except:
                            continue
                    
                    if best_tables:
                        all_tables.extend(best_tables)
                        
                        # Create preview for first table on page
                        for j, table in enumerate(best_tables[:2]):  # Max 2 tables per page
                            if table and len(table) > 0:
                                # Clean table data
                                clean_table = []
                                for row in table:
                                    clean_row = [str(cell).strip() if cell else "" for cell in row]
                                    if any(clean_row):  # Skip completely empty rows
                                        clean_table.append(clean_row)
                                
                                if clean_table:
                                    preview_data = {
                                        'page': i + 1,
                                        'table_index': j + 1,
                                        'total_rows': len(clean_table),
                                        'total_cols': max(len(row) for row in clean_table) if clean_table else 0,
                                        'headers': clean_table[0] if clean_table else [],
                                        'sample_rows': clean_table[1:4] if len(clean_table) > 1 else [],
                                        'extraction_method': "pdfplumber_advanced"
                                    }
                                    table_previews.append(preview_data)
                
                # Analyze content
                word_count = len(all_text.split())
                
                # Enhanced geospatial detection
                geo_indicators = [
                    'latitude', 'longitude', 'lat', 'lon', 'coordinates', 'address',
                    'location', 'city', 'state', 'country', 'zip', 'postal',
                    'geometry', 'polygon', 'point', 'linestring', 'geojson',
                    'projection', 'coordinate system', 'utm', 'wgs84', 'epsg'
                ]
                has_geospatial = any(indicator.lower() in all_text.lower() for indicator in geo_indicators)
                
                return {
                    'type': 'pdf',
                    'pages': page_count,
                    'word_count': word_count,
                    'has_tables': len(all_tables) > 0,
                    'table_count': len(all_tables),
                    'table_previews': table_previews,
                    'text_sample': all_text[:800],
                    'text_heavy': word_count > 500,
                    'mainly_tables': len(all_tables) > 2 and word_count < 200,
                    'has_geospatial': has_geospatial,
                    'file_size_mb': round(filepath.stat().st_size / 1024 / 1024, 2),
                    'extraction_strategies': len(strategies)
                }
                
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def preview_csv_excel(filepath: Path) -> Dict[str, Any]:
        """Preview CSV/Excel files with geospatial detection"""
        try:
            import pandas as pd
            
            # Determine file type and read accordingly
            if filepath.suffix.lower() == '.csv':
                # Try multiple encodings and separators
                encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
                df = None
                
                for encoding in encodings:
                    try:
                        df = pd.read_csv(filepath, encoding=encoding, nrows=1000)  # Sample first 1000 rows
                        break
                    except:
                        continue
                
                if df is None:
                    return {"error": "Could not read CSV with any encoding"}
                    
            else:  # Excel
                try:
                    # Get all sheet names first
                    excel_file = pd.ExcelFile(filepath)
                    sheets = excel_file.sheet_names
                    
                    # Read first sheet sample
                    df = pd.read_excel(filepath, sheet_name=sheets[0], nrows=1000)
                    
                    sheet_info = {"available_sheets": sheets, "current_sheet": sheets[0]}
                except Exception as e:
                    return {"error": f"Excel reading error: {str(e)}"}
            
            # Analyze columns for geospatial content
            geo_columns = []
            for col in df.columns:
                col_lower = col.lower()
                if any(geo_term in col_lower for geo_term in 
                      ['lat', 'lon', 'coord', 'address', 'location', 'city', 'state', 'country', 'zip', 'postal', 'geometry']):
                    geo_columns.append(col)
            
            # Check for coordinate-like data in numeric columns
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            potential_coords = []
            for col in numeric_cols:
                values = df[col].dropna()
                if len(values) > 0:
                    min_val, max_val = values.min(), values.max()
                    # Latitude range: -90 to 90, Longitude range: -180 to 180
                    if -90 <= min_val <= 90 and -90 <= max_val <= 90:
                        potential_coords.append(f"{col} (lat-like)")
                    elif -180 <= min_val <= 180 and -180 <= max_val <= 180:
                        potential_coords.append(f"{col} (lon-like)")
            
            preview_info = {
                "type": "csv" if filepath.suffix.lower() == '.csv' else "excel",
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": list(df.columns),
                "dtypes": df.dtypes.to_dict(),
                "sample_data": df.head(3).to_dict('records'),
                "missing_data": df.isnull().sum().to_dict(),
                "has_geospatial": len(geo_columns) > 0 or len(potential_coords) > 0,
                "geo_columns": geo_columns,
                "potential_coordinates": potential_coords,
                "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2)
            }
            
            if filepath.suffix.lower() != '.csv':
                preview_info.update(sheet_info)
                
            return preview_info
            
        except Exception as e:
            return {"error": str(e)}


class PromptEngineer:
    """Advanced prompt engineering for code generation"""
    
    @staticmethod
    def create_system_prompt() -> str:
        return """You are an expert Python data analyst. Generate clean, robust Python code that:

CRITICAL REQUIREMENTS:
1. Always handle errors with try-except blocks
2. Return results in the EXACT format requested (JSON, base64, numbers, strings)
3. Use standard libraries: pandas, numpy, matplotlib, PIL, json, base64, io, PyPDF2, pdfplumber, docx
4. GEOSPATIAL LIBRARIES: geopandas, shapely, folium, geopy for location-based analysis
5. Never use input() or interactive functions
6. Save plots to BytesIO, convert to base64 as raw string (no data URI prefix)
7. Keep plots under 100KB when possible
8. Handle missing files/data gracefully
9. Use absolute file paths provided in the context
10. ALWAYS convert numpy/pandas types to native Python types before assigning to 'answer'
11. MAKE SURE 'answer' is strictly a JSON object unless specified otherwise in the questions.txt.

GEOSPATIAL ANALYSIS:
- Use geopandas for spatial data manipulation: gpd.read_file(), gpd.GeoDataFrame()
- Use shapely for geometric operations: Point, Polygon, LineString
- Use folium for interactive maps: folium.Map(), folium.Marker()
- Use geopy for geocoding: from geopy.geocoders import Nominatim
- Handle coordinate systems: df.to_crs('EPSG:4326') for WGS84
- For map visualizations, return as base64 HTML or static image
- Example coordinate detection: df['geometry'] = gpd.points_from_xy(df['longitude'], df['latitude'])

IMAGE COMPRESSION:
- Always check image/plot size before returning base64 data
- If base64 image exceeds specified size limits (check questions for size requirements):
  * For matplotlib plots: reduce figure size with plt.figure(figsize=(width, height)), use lower DPI
  * For PIL images: resize with image.resize((new_width, new_height), Image.LANCZOS)
  * Compress JPEG quality: image.save(buffer, format='JPEG', quality=60-80, optimize=True)
  * Convert PNG to JPEG if size reduction needed: image.convert('RGB').save()
  * Make sure that only the final base64 encoded string is printed in response (remove the "data:image/png;base64," part).

PDF PROCESSING:
- First analyze PDF structure with: 
  with pdfplumber.open(filepath) as pdf: 
      page_count = len(pdf.pages)
      first_page_text = pdf.pages[0].extract_text()[:500]
- For TABLE extraction use MULTIPLE strategies:
  * Strategy 1: page.extract_tables(table_settings={{"vertical_strategy": "lines", "horizontal_strategy": "lines"}})
  * Strategy 2: page.extract_tables(table_settings={{"vertical_strategy": "text", "horizontal_strategy": "text"}})  
  * Strategy 3: page.extract_tables(table_settings={{"edge_min_length": 3, "snap_tolerance": 3}})
  * Strategy 4: Try using camelot: camelot.read_pdf(filepath, pages='1', flavor='lattice')
  * Strategy 5: Try using tabula: tabula.read_pdf(filepath, pages=1, multiple_tables=True)
- Try all strategies and use the one with most complete data
- Handle tables that span multiple pages by combining adjacent page tables
- Convert extracted tables to pandas DataFrames immediately: pd.DataFrame(table[1:], columns=table[0])
- Clean table data: handle empty cells, merge duplicate headers, standardize data types

IMAGE PROCESSING WITH OCR:
- Use PIL for image operations: PIL.Image.open(filepath)
- For text extraction from images: import pytesseract; text = pytesseract.image_to_string(image)
- For table extraction from images: try pytesseract with table-specific config
- If any extraction fails or produces null or unknown results, manually answer the questions asked on the image or manually append the context of image as required to answer the questions set provided.
- Handle different image formats (PNG, JPG, GIF, etc.)
- Convert images to base64 with proper data URI format

CSV/EXCEL PROCESSING:
- Use pandas with robust options: pd.read_csv(filepath, encoding='utf-8', on_bad_lines='skip', dtype=str)
- Try multiple encodings if utf-8 fails: ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
- For Excel: specify sheet_name=None to get all sheets, handle merged cells
- Detect and handle geospatial columns automatically

JSON/PARQUET PROCESSING:
- JSON files: Use pd.read_json() or json.load() for complex nested data
- Parquet files: Use pd.read_parquet() for efficient processing
- Handle large files with chunking: pd.read_parquet(filepath, columns=subset_cols)
- For nested JSON: Use pd.json_normalize() to flatten structures

CRITICAL VISUALIZATION RULE:
Strictly match ALL plot or image visualization requirements as described in questions, including colors, font_size, axis labels, plot titles, legends, formats, graph types. For geospatial visualizations, use folium for interactive maps or matplotlib/contextily for static maps.
Make sure the axis are labelled and are visible clearly.
FOR ALL IMAGES/PLOTS: Return ONLY raw base64 string - no data URI prefix.

FINAL ANSWER CONVERSION:
Before assigning to 'answer', always convert numpy/pandas types:
- Use .item() for numpy scalars: answer = result.item() 
- Use .tolist() for numpy arrays: answer = result.tolist()
- Use int()/float() for pandas types: answer = int(result) or float(result)
- Example: answer = float(my_pandas_result) instead of answer = my_pandas_result

CRITICAL API USAGE:
- ONLY use standard library functions and their documented parameters
- If unsure about a function's parameters, use default values or basic syntax
- Do not invent parameter names - stick to commonly known ones
- When using any library function, use the most basic, standard syntax first
- Always handle API errors gracefully with try-except blocks
- Example: Use obj.fit(data) instead of obj.fit(data, unknown_param=value)

ERROR PREVENTION:
- Never use parameters you're not 100% certain exist
- Start with minimal function calls, add parameters only if needed
- Use help() or basic examples for complex functions
- If a function fails, try simpler alternatives or default parameters


OUTPUT FORMAT:
- Return ONLY the Python code, no explanations
- Code should be ready to execute
- Last line should assign result to variable 'answer'
- For images: assign ONLY the raw base64 string to 'answer'
-Unless explicitly mentioned in questions.txt, return the final response strictly as a JSON object with the key 'answer'.
-Make sure the answers (value of keys) to the questions are formatted as per the requirements in the questions.txt file."""

    @staticmethod
    def create_user_prompt(questions_content: str, files_context: str) -> str:
        return f"""QUESTIONS FILE CONTENT:
{questions_content}

AVAILABLE FILES WITH DETAILED PREVIEWS:
{files_context}

Generate Python code that processes ALL questions in the questions file and returns the final answer in the EXACT format requested by the questions file.

IMPORTANT:
- Process all questions in sequence
- Follow the exact output format specified in the questions file (JSON, lists, tables, etc.)
- Assign your final formatted answer to a variable called 'answer'
- For images: return base64 string with proper data URI format
- Handle all possible errors gracefully
- Use the file paths exactly as provided above
- For geospatial data: use geopandas, folium, geopy as needed
- If image/plot size limits are mentioned, implement compression accordingly
- Return the complete response as requested in the questions file

CODE:"""


class GeminiCodeGenerator:
    """Handles Gemini API interactions with advanced prompt engineering"""
    
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        self.prompt_engineer = PromptEngineer()
    
    async def generate_code(self, questions_content: str, files_context: str) -> str:
        """Generate Python code using advanced prompting techniques"""
        
        system_prompt = self.prompt_engineer.create_system_prompt()
        user_prompt = self.prompt_engineer.create_user_prompt(questions_content, files_context)
        
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        try:
            # Use asyncio to handle potential API timeouts
            response = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: self.model.generate_content(full_prompt)
            )
            
            if not response.text:
                raise ValueError("Empty response from Gemini")
                
            # Clean up the response (remove markdown code blocks if present)
            code = response.text.strip()
            if code.startswith('```python'):
                code = code[9:]  # Remove ```python
            if code.startswith('```'):
                code = code[3:]   # Remove ```
            if code.endswith('```'):
                code = code[:-3]  # Remove trailing ```
                
            return code.strip()
            
        except Exception as e:
            raise Exception(f"Gemini API error: {str(e)}")


class CodeExecutor:
    """Safe code execution with timeout and error handling"""
    
    def __init__(self, timeout_seconds: int = 120):  # Increased timeout for large files
        self.timeout = timeout_seconds
    
    async def execute_code(self, code: str, work_dir: Path) -> QuestionResult:
        """Execute generated code safely with comprehensive error handling"""
        
        # Create a secure execution script
        script_content = f"""
import sys
import os
import json
import traceback

# Change to work directory first
os.chdir(r'{work_dir}')

# Now import all the libraries
try:
    import pandas as pd
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    from PIL import Image
    import base64
    from io import BytesIO
    import warnings
    import PyPDF2
    import pdfplumber
    from docx import Document
    import geopandas as gpd
    import shapely
    from shapely.geometry import Point, Polygon, LineString
    import folium
    from geopy.geocoders import Nominatim
    import pytesseract
    import camelot
    import tabula
    import pyarrow.parquet as pq
    warnings.filterwarnings('ignore')
    
    print("Libraries imported successfully")
except ImportError as e:
    print(f"Import error: {{e}}")

# Add error handling wrapper
try:
{self._indent_code(code)}
    
    # Ensure answer variable exists
    if 'answer' not in locals():
        answer = "No answer variable found"
        
    # Convert answer to JSON-serializable format
    if hasattr(answer, 'tolist'):  # numpy arrays
        answer = answer.tolist()
    elif hasattr(answer, 'to_dict'):  # pandas objects
        answer = answer.to_dict()
    print("RESULT_START")
    if isinstance(answer, (dict, list)):
        print(json.dumps(answer))  # Keep this for proper JSON formatting
    else:
        print(json.dumps(answer))
    print("RESULT_END")
    
except Exception as e:
    print("ERROR_START")
    print(f"Error: {{str(e)}}")
    print(f"Traceback: {{traceback.format_exc()}}")
    print("ERROR_END")
"""
        
        # Write script to temporary file
        script_path = work_dir / "execution_script.py"
        script_path.write_text(script_content)
        
        try:
            # Execute with timeout using uv run
            process = await asyncio.create_subprocess_exec(
                'uv', 'run', 'python', str(script_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=work_dir
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), 
                    timeout=self.timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                raise Exception(f"Code execution timed out after {self.timeout} seconds")
            
            # Parse output
            output = stdout.decode('utf-8')
            error_output = stderr.decode('utf-8')
            
            # Extract result or error
            if "RESULT_START" in output and "RESULT_END" in output:
                result_start = output.find("RESULT_START") + len("RESULT_START\n")
                result_end = output.find("RESULT_END")
                result_json = output[result_start:result_end].strip()
                
                try:
                    result = json.loads(result_json)
                    # If result is a string that looks like JSON, parse it again
                    if isinstance(result, str) and result.strip().startswith(('{', '[')):
                        result = json.loads(result)
                    return QuestionResult(
                        question_number=0,
                        question_text="",
                        result=result,
                        success=True
                    )
                except json.JSONDecodeError:
                    # If JSON parsing fails, return as string
                    return QuestionResult(
                        question_number=0,
                        question_text="",
                        result=result_json,
                        success=True
    )
            
            elif "ERROR_START" in output and "ERROR_END" in output:
                error_start = output.find("ERROR_START") + len("ERROR_START\n")
                error_end = output.find("ERROR_END")
                error_message = output[error_start:error_end].strip()
                
                return QuestionResult(
                    question_number=0,
                    question_text="",
                    result="could not answer",
                    success=False,
                    error_message=error_message
                )
            
            else:
                # Fallback: process stderr or stdout
                error_info = error_output if error_output else "Unknown execution error"
                return QuestionResult(
                    question_number=0,
                    question_text="",
                    result="could not answer",
                    success=False,
                    error_message=error_info
                )
                
        except Exception as e:
            return QuestionResult(
                question_number=0,
                question_text="",
                result="could not answer",
                success=False,
                error_message=str(e)
            )
        finally:
            # Cleanup
            if script_path.exists():
                script_path.unlink()
    
    def _indent_code(self, code: str, spaces: int = 4) -> str:
        """Indent code for proper Python execution"""
        return '\n'.join(' ' * spaces + line for line in code.split('\n'))


class QuestionParser:
    """Parse and structure questions from questions.txt"""
    
    @staticmethod
    def parse_questions(content: str) -> List[str]:
        """Extract individual questions from questions.txt content"""
        
        # Remove the instruction part and focus on numbered questions
        lines = content.strip().split('\n')
        questions = []
        current_question = ""
        
        for line in lines:
            line = line.strip()
            
            # Check if line starts with a number (question number)
            if re.match(r'^\d+\.', line):
                # Save previous question if exists
                if current_question:
                    questions.append(current_question.strip())
                
                # Start new question (remove the number prefix)
                current_question = re.sub(r'^\d+\.\s*', '', line)
            
            elif current_question and line:  # Continue current question
                current_question += " " + line
        
        # Add the last question
        if current_question:
            questions.append(current_question.strip())
        
        return questions


class DataAnalystAgent:
    """Main orchestrator for the data analysis agent"""
    
    def __init__(self, gemini_api_key: str):
        self.code_generator = GeminiCodeGenerator(gemini_api_key)
        self.code_executor = CodeExecutor()
        self.question_parser = QuestionParser()
        self.url_extractor = URLExtractor()
        self.scraper_client = ScraperClient()
        self.preview_generator = FilePreviewGenerator()
        
        # Initialize formatting model (Gemini 1.5 Flash)
        self.formatting_model = genai.GenerativeModel('gemini-1.5-flash')
    
    def _extract_images(self, answer: Any, start_counter: int = 1) -> Tuple[Any, Dict[str, str]]:
        
        """Extract base64 images and replace with placeholders"""
        images = {}
        image_counter = start_counter
        
        def process_value(value):
            nonlocal image_counter
            if isinstance(value, str):
                # Pattern to match data URI images
                pattern = r'data:image/[^;]+;base64,([A-Za-z0-9+/=]+)'
                matches = list(re.finditer(pattern, value))
                
                processed_value = value
                for match in matches:
                    base64_data = match.group(1)
                    full_match = match.group(0)
                    placeholder = f"{{{{IMAGE_{image_counter}}}}}"
                    images[placeholder] = base64_data
                    processed_value = processed_value.replace(full_match, placeholder, 1)
                    image_counter += 1
                
                return processed_value
            elif isinstance(value, dict):
                return {k: process_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [process_value(item) for item in value]
            else:
                return value
        
        processed_answer = process_value(answer)
        return processed_answer, images
    
    def _restore_images(self, formatted_response: Any, image_map: Dict[str, str]) -> Any:
        """Restore base64 images from placeholders"""
        if not isinstance(formatted_response, str) or not image_map:
            return formatted_response
        
        result = formatted_response
        for placeholder, base64_data in image_map.items():
            print(f"📄 Replacing {placeholder} with image data ({len(base64_data)} chars)")
            old_result = result
            result = result.replace(placeholder, base64_data)
            replacements = old_result.count(placeholder)
            print(f"   Made {replacements} replacements")
        
        return result

    async def _call_formatting_llm(self, questions_content: str, processed_answers: List[Any]) -> str:
        """Call Gemini 1.5 Flash to format the final response"""
        
        system_prompt = """You are an expert response formatter. Your job is to take raw answers from a data analysis system and format them according to the specific instructions in the questions file.

CRITICAL FORMATTING RULES:
1. Follow the EXACT format requested in each question (JSON, tables, lists, paragraphs, etc.)
2. Maintain the same order as the original questions
3. If a question asks for specific units, include them
4. If a question asks for rounded numbers, apply the rounding
5. If a question asks for a specific number of items, limit to that count
6. Preserve any {{IMAGE_N}} placeholders exactly as they are
7. If an answer is "could not answer", state this clearly but professionally
8. For numerical answers, use the exact precision requested
9. For text answers, use the exact style requested (bullet points, paragraphs, etc.)
10. If questions ask for explanations along with answers, provide both

CRITICAL JSON FORMATTING RULES:
1. Return ONLY valid, complete JSON - no extra text before or after
2. Ensure all JSON braces/brackets are properly matched
3. Do not include explanations, comments, or additional content outside the JSON
4. Make sure your json response can be extracted by using json.loads(response) on the backend

RESPONSE FORMAT:
- Return answers in the same order as questions
- Separate each answer clearly
- Use professional, clear language
- Be concise but complete
- Follow any specific formatting mentioned in questions.txt

Remember: You are NOT analyzing data yourself - you are formatting pre-analyzed results to match the requested output format."""

        user_prompt = f"""QUESTIONS FROM FILE:
{questions_content}

RAW ANSWERS FROM ANALYSIS:
{json.dumps(processed_answers, indent=2)}

Please format these answers according to the specific requirements in the questions file. Maintain the order and follow the exact format requested for each question. Keep any {{{{IMAGE_N}}}} placeholders intact.
Give only the formatted response, no explanations or additional text."""

        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.formatting_model.generate_content(
                    f"{system_prompt}\n\n{user_prompt}",
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.1,  # Low temperature for consistent formatting
                        max_output_tokens=65536,
                    )
                )
            )
            
            if response.text:
                return response.text.strip()
            else:
                raise Exception("Empty response from formatting LLM")
                
        except Exception as e:
            print(f"⚠️ Formatting LLM error: {e}")
            return json.dumps(processed_answers)  # Fallback to original

    async def format_final_response(self, questions_content: str, raw_answers: Any) -> Any:
        """Format the final response using LLM with image placeholder handling"""
        try:
            # Handle single scraper response vs list of answers
            # Single response from scraper - treat as single answer
            processed_answers = [raw_answers]
            """else:
                # Original list of answers from individual question processing
                processed_answers = []
                all_images = {}
                image_counter = 1
                
                for i, answer in enumerate(raw_answers):
                    processed_answer, images = self._extract_images(answer, image_counter)
                    processed_answers.append(processed_answer)
                    all_images.update(images)
                    image_counter += len(images)
                
                print(f"🖼️ Extracted {len(all_images)} images for formatting")
                
                # Call formatting LLM
                formatted_response = await self._call_formatting_llm(questions_content, processed_answers)
                print(f"🤖 LLM returned placeholders:")
                for placeholder in all_images.keys():
                    count = formatted_response.count(placeholder)
                    print(f"  {placeholder}: appears {count} times")
                print(f"📋 Image map contents:")
                image_data_seen = {}
                for placeholder, base64_data in all_images.items():
                    short_hash = base64_data[:50] + "..." if len(base64_data) > 50 else base64_data
                    print(f"  {placeholder}: {short_hash}")
                    
                    # Check for duplicate base64 data
                    if base64_data in image_data_seen:
                        print(f"  ⚠️ DUPLICATE IMAGE DATA! Same as {image_data_seen[base64_data]}")
                    else:
                        image_data_seen[base64_data] = placeholder

                # Restore images in formatted response
                final_response = self._restore_images(formatted_response, all_images)
                if isinstance(final_response, str):
                    if final_response.startswith("```json"):
                        final_response = final_response[len("```json\n"):]
                    if final_response.endswith("```"):
                        final_response = final_response[:-3]
                    final_response = final_response.strip()
                    final_response=final_response.replace("\n", "")
                    print(final_response)
                    if final_response.startswith('"') and final_response.endswith('"'):
                        try:
                            # Remove outer quotes and unescape
                            final_response = json.loads(final_response)
                        except json.JSONDecodeError:
                            pass
                    # Only parse if it looks like JSON and handle parsing errors
                    if final_response.startswith(('{', '[')):
                        try:
                            final_response = json.loads(final_response)
                        except json.JSONDecodeError as e:
                            print(f"⚠️ JSON parsing failed: {e}")

                return final_response"""
            
            # For single scraper response, extract images and process
            all_images = {}
            processed_answer, images = self._extract_images(processed_answers[0])
            processed_answers = [processed_answer]
            all_images.update(images)

            print(f"🖼️ Extracted {len(all_images)} images for formatting")

            # Call formatting LLM
            formatted_response = await self._call_formatting_llm(questions_content, processed_answers)

            # Restore images in formatted response
            final_response = formatted_response

            if isinstance(final_response, str):
                if final_response.startswith("```json"):
                    final_response = final_response[len("```json\n"):]
                if final_response.endswith("```"):
                    final_response = final_response[:-3]
                
                final_response = final_response.strip()
                
                # Restore images BEFORE JSON parsing
                final_response = self._restore_images(final_response, all_images)
                
                # Only parse if it looks like JSON and handle parsing errors
                if final_response.startswith(('{', '[')):
                    try:
                        final_response = json.loads(final_response)
                    except json.JSONDecodeError as e:
                        print(f"⚠️ JSON parsing failed: {e}")
                        pass
                elif final_response.startswith('"') and final_response.endswith('"'):
                    # For quoted strings, use json.loads to handle escaping properly
                    try:
                        final_response = json.loads(final_response)
                    except json.JSONDecodeError as e:
                        print(f"⚠️ JSON parsing failed: {e}")
                        # Remove quotes manually as fallback
                        final_response = final_response[1:-1]

            return final_response
            
        except Exception as e:
            print(f"⚠️ Response formatting failed: {e}")
            return raw_answers 

    async def process_request(self, questions_content: str, uploaded_files: Dict[str, Path]) -> Dict[str, Any]:
        """Process the complete request and return structured results"""
        
        # Wake up scraper service (always, regardless of whether we need it)
        asyncio.create_task(self.scraper_client.wake_up_scraper())
        
        # Check if URLs exist and if only scraping is needed
        urls = self.url_extractor.extract_urls(questions_content)
        other_files_exist = len(uploaded_files) > 1 or (len(uploaded_files) == 1 and 'questions.txt' not in uploaded_files and 'question.txt' not in uploaded_files)
        
        # If URLs exist and no other files, do scraping only
        if urls and not other_files_exist:
            print(f"🔗 Found {len(urls)} URLs, scraping only mode...")
            try:
                scraper_response = await self.scraper_client.scrape_with_questions(questions_content)
                print("✅ Scraping completed, returning response...")
                return await self.format_scraper_response(questions_content, scraper_response)
                
            except Exception as e:
                print(f"❌ Scraping failed: {e}")
                return {"error": f"Scraping failed: {str(e)}"}
        
        # Create enhanced files context with previews
        files_context = await self._create_enhanced_files_context(uploaded_files)
        
        # Process all questions in single prompt
        print(f"Processing all questions in single prompt...")
        
        try:
            # Generate code for all questions
            print(f"Generating code for all questions...")
            code = await self.code_generator.generate_code(questions_content, files_context)
            print(f"Generated code length: {len(code)} characters")
            print(f"Code preview: {code[:200]}...")
            
            # Execute code
            work_dir = list(uploaded_files.values())[0].parent if uploaded_files else Path.cwd()
            result = await self.code_executor.execute_code(code, work_dir)
            
            if not result.success:
                print(f"Code execution error: {result.error_message}")
                return {"error": f"Code execution failed: {result.error_message}"}
            
            print(f"All questions completed: ✅")
            
            # Return the result directly (no additional formatting needed)
            return result.result
            
        except Exception as e:
            print(f"Processing failed with exception: {str(e)}")
            import traceback
            traceback.print_exc()
            return {"error": f"Processing failed: {str(e)}"}

    async def _create_enhanced_files_context(self, uploaded_files: Dict[str, Path]) -> str:
        """Create enhanced context string with detailed file previews"""
        
        if not uploaded_files:
            return "No files available."
        
        context_lines = ["Available files with detailed analysis:"]
        
        for filename, filepath in uploaded_files.items():
            file_size = filepath.stat().st_size
            file_ext = filepath.suffix.lower()
            
            context_lines.append(f"\n📁 {filename}")
            context_lines.append(f"   Path: '{filepath}'")
            context_lines.append(f"   Size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
            context_lines.append(f"   Type: {file_ext}")
            
            # Generate detailed preview based on file type
            try:
                if file_ext == '.csv' or file_ext in ['.xls', '.xlsx']:
                    preview = self.preview_generator.preview_csv_excel(filepath)
                    if 'error' not in preview:
                        context_lines.append(f"   📊 {preview['type'].upper()} DATA:")
                        context_lines.append(f"   ├─ Rows: {preview['rows']:,}")
                        context_lines.append(f"   ├─ Columns: {preview['columns']}")
                        context_lines.append(f"   ├─ Column Names: {preview['column_names'][:10]}{'...' if len(preview['column_names']) > 10 else ''}")
                        context_lines.append(f"   ├─ Data Types: {list(preview['dtypes'].keys())[:5]}{'...' if len(preview['dtypes']) > 5 else ''}")
                        
                        if preview.get('has_geospatial'):
                            context_lines.append(f"   🌍 GEOSPATIAL DETECTED:")
                            if preview.get('geo_columns'):
                                context_lines.append(f"       ├─ Geographic columns: {preview['geo_columns']}")
                            if preview.get('potential_coordinates'):
                                context_lines.append(f"       └─ Potential coordinates: {preview['potential_coordinates']}")
                        
                        if preview.get('available_sheets'):
                            context_lines.append(f"   └─ Excel sheets: {preview['available_sheets']}")
                    else:
                        context_lines.append(f"   ❌ Preview error: {preview['error']}")
                
                elif file_ext == '.json':
                    preview = self.preview_generator.preview_json(filepath)
                    if 'error' not in preview:
                        context_lines.append(f"   📋 JSON DATA:")
                        if preview['type'] == 'json_object':
                            context_lines.append(f"   ├─ Type: Object with {preview['total_keys']} keys")
                            context_lines.append(f"   ├─ Keys: {preview['keys'][:10]}{'...' if len(preview['keys']) > 10 else ''}")
                            context_lines.append(f"   ├─ Structure: {preview['structure']}")
                            context_lines.append(f"   └─ Sample data: {preview['sample_data']}")
                        elif preview['type'] == 'json_array':
                            context_lines.append(f"   ├─ Type: Array with {preview['length']} items")
                            context_lines.append(f"   ├─ Item types: {preview['item_types']}")
                            context_lines.append(f"   └─ Sample: {preview['sample_items']}")
                        else:
                            context_lines.append(f"   └─ Format: {preview['format']}")
                    else:
                        context_lines.append(f"   ❌ Preview error: {preview['error']}")
                
                elif file_ext == '.parquet':
                    preview = self.preview_generator.preview_parquet(filepath)
                    if 'error' not in preview:
                        context_lines.append(f"   🗂️ PARQUET DATA:")
                        context_lines.append(f"   ├─ Rows: {preview['rows']:,}")
                        context_lines.append(f"   ├─ Columns: {preview['columns']}")
                        context_lines.append(f"   ├─ Column info: {list(preview['column_info'].keys())[:8]}{'...' if len(preview['column_info']) > 8 else ''}")
                        context_lines.append(f"   ├─ Memory usage: {preview['memory_usage']}")
                        context_lines.append(f"   └─ Sample: {len(preview['sample_data'])} rows preview available")
                    else:
                        context_lines.append(f"   ❌ Preview error: {preview['error']}")
                
                elif file_ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
                    preview = self.preview_generator.preview_image(filepath)
                    if 'error' not in preview:
                        context_lines.append(f"   🖼️ IMAGE DATA:")
                        context_lines.append(f"   ├─ Format: {preview['format']}")
                        context_lines.append(f"   ├─ Size: {preview['size']} pixels ({preview['megapixels']} MP)")
                        context_lines.append(f"   ├─ Mode: {preview['mode']}")
                        
                        if preview.get('has_text'):
                            context_lines.append(f"   ├─ Contains text: YES")
                            context_lines.append(f"   ├─ Text sample: {preview['text_sample'][:100]}...")
                            if preview.get('likely_table'):
                                context_lines.append(f"   └─ 📊 Likely contains tabular data - use OCR extraction")
                        else:
                            context_lines.append(f"   └─ Contains text: NO")
                        
                        context_lines.append(f"   💡 Use PIL.Image.open() and pytesseract for text extraction")
                    else:
                        context_lines.append(f"   ❌ Preview error: {preview['error']}")
                
                elif file_ext == '.pdf':
                    preview = self.preview_generator.preview_pdf(filepath)
                    if 'error' not in preview:
                        context_lines.append(f"   📄 PDF ANALYSIS:")
                        context_lines.append(f"   ├─ Pages: {preview['pages']}")
                        context_lines.append(f"   ├─ Size: {preview['file_size_mb']} MB")
                        context_lines.append(f"   ├─ Word count: {preview['word_count']:,}")
                        context_lines.append(f"   ├─ Content type: {'Text-heavy' if preview['text_heavy'] else 'Table-heavy' if preview['mainly_tables'] else 'Mixed'}")
                        
                        if preview['has_tables']:
                            context_lines.append(f"   ├─ 📊 TABLES FOUND: {preview['table_count']} sections")
                            for i, table_preview in enumerate(preview['table_previews'][:3]):  # Show max 3 previews
                                context_lines.append(f"   │   Table {i+1} (Page {table_preview['page']}): {table_preview['total_rows']} rows × {table_preview['total_cols']} cols")
                                if table_preview['headers']:
                                    headers = ' | '.join(str(h)[:12] for h in table_preview['headers'][:5])
                                    context_lines.append(f"   │   Headers: {headers}{'...' if len(table_preview['headers']) > 5 else ''}")
                            
                            context_lines.append(f"   ├─ 🔧 Use pdfplumber with multiple extraction strategies")
                            context_lines.append(f"   │   Also try camelot and tabula as fallbacks")
                        
                        if preview.get('has_geospatial'):
                            context_lines.append(f"   ├─ 🌍 GEOSPATIAL CONTENT DETECTED")
                            context_lines.append(f"   │   May contain coordinates, addresses, or location data")
                        
                        context_lines.append(f"   └─ Text sample: {preview['text_sample'][:150].replace(chr(10), ' ')}...")
                    else:
                        context_lines.append(f"   ❌ Preview error: {preview['error']}")
                        
                elif file_ext in ['.doc', '.docx']:
                    context_lines.append(f"   📝 WORD DOCUMENT")
                    context_lines.append(f"   └─ 💡 Use python-docx Document() to load and extract tables/text")
                    
                elif file_ext == '.txt':
                    # Quick text preview
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            text_sample = f.read(1000)
                        context_lines.append(f"   📄 TEXT FILE")
                        context_lines.append(f"   ├─ Sample: {text_sample[:200].replace(chr(10), ' ')}...")
                        context_lines.append(f"   └─ Use open() to read full content")
                    except:
                        context_lines.append(f"   📄 TEXT FILE - Use open() to read content")
                
                else:
                    context_lines.append(f"   🔍 GENERIC FILE - Determine processing method based on content")
                    
            except Exception as e:
                context_lines.append(f"   ❌ Preview generation failed: {str(e)}")
        
        return '\n'.join(context_lines)
    
    async def format_scraper_response(self, questions_content: str, scraper_response: Dict[str, Any]) -> Any:
        """Format scraper response using the existing formatting pipeline"""
       
        
        try:
            # Extract the actual analysis from scraper response structure
            raw_result = scraper_response.get('result', {}).get('results', {})
            
            # Use existing format_final_response (it handles image extraction internally)
            return await self.format_final_response(questions_content, raw_result)
            
        except Exception as e:
            print(f"⚠️ Scraper response formatting failed: {e}")
            return scraper_response


# FastAPI Application
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    print("🚀 Data Analyst Agent starting up...")
    yield
    print("📊 Data Analyst Agent shutting down...")


app = FastAPI(
    title="Data Analyst Agent",
    description="AI-powered data analysis agent that generates and executes code to answer questions",
    version="1.0.0",
    lifespan=lifespan
)


from fastapi import Request

@app.post("/api")
async def analyze_data(request: Request):
    """Main API endpoint for data analysis - accepts dynamic file fields"""
    
    # Get Gemini API key from environment
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500, 
            detail="GEMINI_API_KEY environment variable not set"
        )
    
    # Parse form data to get files dynamically
    form_data = await request.form()
    
    if not form_data:
        raise HTTPException(
            status_code=400,
            detail="No form data received"
        )
    
    # Create temporary directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        uploaded_files = {}
        questions_content = None
        
        try:
            # Process each field in the form data
            for field_name, field_value in form_data.items():
                if hasattr(field_value, 'filename') and hasattr(field_value, 'read'):
                    # This is a file upload
                    filename = field_value.filename or field_name
                    file_path = temp_path / filename
                    
                    # Read and save file content
                    content = await field_value.read()
                    file_path.write_bytes(content)
                    
                    uploaded_files[filename] = file_path
                    
                    # Extract questions.txt content
                    if filename.lower() == 'questions.txt' or filename.lower()=='question.txt':
                        questions_content = content.decode('utf-8')
            
            if not questions_content:
                raise HTTPException(
                    status_code=400, 
                    detail="questions.txt file is required"
                )
            
            if not uploaded_files:
                raise HTTPException(
                    status_code=400,
                    detail="No valid files uploaded"
                )
            
            print(f"Received files: {list(uploaded_files.keys())}")
            print(f"Questions preview: {questions_content[:200]}...")
            
            # Initialize and run agent
            agent = DataAnalystAgent(api_key)
            results = await agent.process_request(questions_content, uploaded_files)
            
            return JSONResponse(content=results)
            
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback; traceback.print_exc() 
            raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "data-analyst-agent"}


if __name__ == "__main__":
    # For development
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )