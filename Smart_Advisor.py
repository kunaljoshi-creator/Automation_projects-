import streamlit as st
import pdfplumber
from pdf2image import convert_from_bytes
import pytesseract
from PIL import Image

# Configure path for Windows
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# Function to extract text using pdfplumber or fallback to OCR
def extract_text_from_pdf(pdf_file):
    text = ""
    ocr_needed = False

    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted
                else:
                    ocr_needed = True
    except:
        ocr_needed = True

    # Rewind file pointer for OCR if needed
    pdf_file.seek(0)
    if not text.strip() or ocr_needed:
        text = extract_text_with_ocr(pdf_file)

    return text.strip()


# Enhanced OCR text extraction using pytesseract and preprocessing
def extract_text_with_ocr(pdf_file):
    text = ""
    images = convert_from_bytes(pdf_file.read(), dpi=300)

    for image in images:
        # Convert to grayscale
        gray = image.convert("L")

        # Thresholding
        threshold = gray.point(lambda x: 0 if x < 150 else 255, '1')

        # Resize
        resized = threshold.resize(
            (threshold.width * 2, threshold.height * 2),
            Image.LANCZOS
        )

        # OCR with Marathi, Hindi, English support
        ocr_text = pytesseract.image_to_string(
            resized,
            lang='mar+hin+eng',
            config='--psm 6'
        )

        text += ocr_text + "\n"

    return text


# Gemini AI summarization with language support
def online_summarize_text(text, language):
    try:
        import google.generativeai as genai
        genai.configure(api_key="Please Enter Your API Key")
        model = genai.GenerativeModel("gemini-1.5-flash")

        prompt = f"Please summarize the following text in {language}:\n\n{text}"
        response = model.generate_content([prompt])
        return response.text
    except Exception as e:
        return f"An error occurred: {e}"


# Gemini AI question-answering with language understanding
def online_question_text(text, question, language):
    try:
        import google.generativeai as genai
        genai.configure(api_key="please enter your API key")
        model = genai.GenerativeModel("gemini-1.5-flash")

        prompt = (
            f"Please answer the following question based on the provided text. "
            f"The answer should be in {language}.\n\n"
            f"Text: {text}\n\nQuestion: {question}"
        )
        response = model.generate_content([prompt])
        return response.text
    except Exception as e:
        return f"An error occurred: {e}"


# Streamlit app
def main():
    st.title("Smart Advisor: The Ultimate Text Q&A and Summarizer 🧠")
    st.markdown("*Powered by Google Gemini AI*")

    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    # Language selection
    language = st.selectbox("Choose output language", ["English", "Hindi", "Marathi"])

    if uploaded_file is not None:
        with st.spinner("Extracting text..."):
            text = extract_text_from_pdf(uploaded_file)

        display_text = text[:500] + ('...' if len(text) > 500 else '')
        st.subheader("Extracted Text")
        st.text_area("Text from PDF", display_text, height=300)

        if st.button("Get Summary"):
            summary = online_summarize_text(text, language)
            st.subheader("Summary")
            st.write(summary)

        question = st.text_input("Enter your question about the text")
        if st.button("Get Answer"):
            if question:
                answer = online_question_text(text, question, language)
                st.subheader("Answer")
                st.write(answer)
            else:
                st.warning("Please enter a question to get an answer.")


if __name__ == "__main__":
    main()