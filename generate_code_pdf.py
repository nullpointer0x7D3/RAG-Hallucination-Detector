from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, PreformattedText
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import os

def create_code_pdf():
    # Create PDF document
    doc = SimpleDocTemplate("Medical_RAG_Code_Documentation.pdf", pagesize=A4)
    story = []
    
    # Get styles
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=colors.darkblue
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=18,
        spaceAfter=20,
        spaceBefore=20,
        textColor=colors.darkred
    )
    
    code_style = ParagraphStyle(
        'CodeStyle',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=9,
        leftIndent=20,
        rightIndent=20,
        spaceAfter=10
    )
    
    # Title page
    story.append(Paragraph("Medical RAG Project", title_style))
    story.append(Spacer(1, 20))
    story.append(Paragraph("Code Documentation", styles['Heading2']))
    story.append(Spacer(1, 20))
    story.append(Paragraph("A comprehensive guide to the Medical RAG system using Meditron 7B LLM, Qdrant Vector Database, and PubMedBERT Embedding Model.", styles['Normal']))
    story.append(Spacer(1, 30))
    story.append(Paragraph("Project Overview:", styles['Heading3']))
    story.append(Paragraph("This project implements a Medical Question-Answering system using Retrieval-Augmented Generation (RAG) with the following components:", styles['Normal']))
    story.append(Spacer(1, 10))
    story.append(Paragraph("• Meditron 7B LLM for medical knowledge generation", styles['Normal']))
    story.append(Paragraph("• Qdrant Vector Database for document storage and retrieval", styles['Normal']))
    story.append(Paragraph("• PubMedBERT embeddings for medical document understanding", styles['Normal']))
    story.append(Paragraph("• FastAPI web interface for user interaction", styles['Normal']))
    story.append(PageBreak())
    
    # Table of Contents
    story.append(Paragraph("Table of Contents", heading_style))
    story.append(Spacer(1, 20))
    story.append(Paragraph("1. rag.py - Main FastAPI Application", styles['Normal']))
    story.append(Paragraph("2. ingest.py - Document Ingestion and Vector Database Creation", styles['Normal']))
    story.append(Paragraph("3. retriever.py - Document Retrieval Testing", styles['Normal']))
    story.append(Paragraph("4. requirements.txt - Dependencies", styles['Normal']))
    story.append(Paragraph("5. README.md - Project Description", styles['Normal']))
    story.append(PageBreak())
    
    # 1. rag.py
    story.append(Paragraph("1. rag.py - Main FastAPI Application", heading_style))
    story.append(Paragraph("This is the core application file that implements the FastAPI web server and RAG functionality.", styles['Normal']))
    story.append(Spacer(1, 10))
    
    with open('rag.py', 'r', encoding='utf-8') as f:
        rag_code = f.read()
    
    story.append(PreformattedText(rag_code, code_style))
    story.append(PageBreak())
    
    # 2. ingest.py
    story.append(Paragraph("2. ingest.py - Document Ingestion and Vector Database Creation", heading_style))
    story.append(Paragraph("This script handles the ingestion of PDF documents, text chunking, and creation of the vector database.", styles['Normal']))
    story.append(Spacer(1, 10))
    
    with open('ingest.py', 'r', encoding='utf-8') as f:
        ingest_code = f.read()
    
    story.append(PreformattedText(ingest_code, code_style))
    story.append(PageBreak())
    
    # 3. retriever.py
    story.append(Paragraph("3. retriever.py - Document Retrieval Testing", heading_style))
    story.append(Paragraph("This script provides a simple interface to test document retrieval from the vector database.", styles['Normal']))
    story.append(Spacer(1, 10))
    
    with open('retriever.py', 'r', encoding='utf-8') as f:
        retriever_code = f.read()
    
    story.append(PreformattedText(retriever_code, code_style))
    story.append(PageBreak())
    
    # 4. requirements.txt
    story.append(Paragraph("4. requirements.txt - Dependencies", heading_style))
    story.append(Paragraph("List of Python packages required to run the Medical RAG system.", styles['Normal']))
    story.append(Spacer(1, 10))
    
    with open('requirements.txt', 'r', encoding='utf-8') as f:
        requirements_content = f.read()
    
    story.append(PreformattedText(requirements_content, code_style))
    story.append(PageBreak())
    
    # 5. README.md
    story.append(Paragraph("5. README.md - Project Description", heading_style))
    story.append(Paragraph("Project overview and description.", styles['Normal']))
    story.append(Spacer(1, 10))
    
    with open('README.md', 'r', encoding='utf-8') as f:
        readme_content = f.read()
    
    story.append(PreformattedText(readme_content, code_style))
    story.append(PageBreak())
    
    # Setup and Usage Instructions
    story.append(Paragraph("Setup and Usage Instructions", heading_style))
    story.append(Spacer(1, 20))
    
    story.append(Paragraph("Prerequisites:", styles['Heading3']))
    story.append(Paragraph("• Python 3.8+ installed", styles['Normal']))
    story.append(Paragraph("• Qdrant vector database running on localhost:6333", styles['Normal']))
    story.append(Paragraph("• Meditron 7B GGUF model file (meditron-7b.Q4_K_M.gguf)", styles['Normal']))
    story.append(Spacer(1, 15))
    
    story.append(Paragraph("Installation:", styles['Heading3']))
    story.append(Paragraph("1. Install dependencies: pip install -r requirements.txt", styles['Normal']))
    story.append(Paragraph("2. Start Qdrant: qdrant (or docker run -p 6333:6333 qdrant/qdrant)", styles['Normal']))
    story.append(Paragraph("3. Ingest documents: python ingest.py", styles['Normal']))
    story.append(Paragraph("4. Run the application: uvicorn rag:app --host 0.0.0.0 --port 8000", styles['Normal']))
    story.append(Spacer(1, 15))
    
    story.append(Paragraph("Architecture:", styles['Heading3']))
    story.append(Paragraph("The system follows a typical RAG architecture:", styles['Normal']))
    story.append(Paragraph("• Document ingestion and chunking (ingest.py)", styles['Normal']))
    story.append(Paragraph("• Vector embedding and storage (Qdrant)", styles['Normal']))
    story.append(Paragraph("• Query processing and retrieval (retriever.py)", styles['Normal']))
    story.append(Paragraph("• LLM-based answer generation (rag.py)", styles['Normal']))
    story.append(Paragraph("• Web interface for user interaction (FastAPI)", styles['Normal']))
    
    # Build PDF
    doc.build(story)
    print("PDF generated successfully: Medical_RAG_Code_Documentation.pdf")

if __name__ == "__main__":
    create_code_pdf()
