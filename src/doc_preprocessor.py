from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from src.config.logger import get_logger

logger = get_logger(__name__)


def convert_to_docling(source: str, use_ocr: bool = False):
    """
    Convert a document to Docling format.
    
    Args:
        source: Path to the document file
        use_ocr: Whether to use OCR for scanned documents (default: False for speed)
    
    Returns:
        Docling document object or None if conversion fails
    """
    try:
        logger.info(f"Attempting to convert document: {source}")
        
        # Configure pipeline options
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = use_ocr  # Disable OCR by default for speed
        
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        
        if source:
            doc = converter.convert(source).document
            logger.info("Successful conversion")
            return doc
            
    except Exception as e:
        logger.error(f"Error while converting: {str(e)}")
        return None