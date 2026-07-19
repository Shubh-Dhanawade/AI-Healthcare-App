import os
import sys

# Ensure backend directory is in the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from app.services.ocr_service import extract_text_from_image
except ImportError as e:
    print(f"Error importing ocr_service: {e}", file=sys.stderr)
    sys.exit(1)

def main():
    if len(sys.argv) < 2:
        print("Error: No input file path provided.", file=sys.stderr)
        sys.exit(1)
        
    file_path = sys.argv[1]
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}", file=sys.stderr)
        sys.exit(1)
        
    try:
        text, method, page_count = extract_text_from_image(file_path)
        # Output the extracted text to stdout
        sys.stdout.write(text)
    except Exception as e:
        print(f"Error executing OCR: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
