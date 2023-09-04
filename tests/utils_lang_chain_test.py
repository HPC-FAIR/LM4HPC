from lm4hpc._utils_langchain import get_pdf_text, get_chunk_text
import os

def test_get_pdf_text():
    """
    Test the get_pdf_text function.
    """
    # Test a single PDF file
    pdf_file = "lm4hpc/data/openmp_pdfs/OMP_official_merged copy.pdf"
    text = get_pdf_text(pdf_file)
    assert len(text) == 1455111

    # # Test a directory of PDF files
    # pdf_dir = os.path.join(os.path.dirname(__file__), 'data/openmp_pdfs')
    # text = get_pdf_text(pdf_dir)
    # assert len(text) == 1455111 * 2

if __name__ == '__main__':
    test_get_pdf_text()