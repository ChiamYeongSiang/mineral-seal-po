import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import tempfile

# Your OpenAI API key
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

def extract_text_from_pdf(uploaded_file):
    from pypdf import PdfReader
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def load_po(file_path):
    loader = TextLoader(file_path)
    documents = loader.load()
    return documents[0].page_content

def extract_po_info(po_text):
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    prompt = ChatPromptTemplate.from_template("""
    You are an AI assistant helping to process purchase orders for Mineral Seal Corporation.
    Mineral Seal Corporation is the SELLER receiving this purchase order.
    The CUSTOMER is the company sending the purchase order — they appear in the FROM field, not the TO field.
    
    Extract the following information from this purchase order and return it in a clear format:
    - Customer Name
    - PO Number
    - Date
    - Ship To Address
    - Line Items (description, quantity, unit price)
    - Shipping Method
    - Payment Terms
    - Special Instructions
    
    Purchase Order:
    {po_text}
    
    Return the extracted information in a clear structured format.
    """)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"po_text": po_text})

def extract_line_items(po_text):
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    prompt = ChatPromptTemplate.from_template("""
    Extract only the line item descriptions from this purchase order.
    Return each item description on a new line, nothing else.
    Do not include quantities, prices, or other details.
    
    Purchase Order:
    {po_text}
    """)
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"po_text": po_text})
    items = [line.strip() for line in result.strip().split('\n') if line.strip()]
    return items

def is_industrial_sealing_product(description):
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    prompt = ChatPromptTemplate.from_template("""
    Is this product description related to industrial sealing, insulating, gasket, packing, or textile materials?
    Answer only YES or NO. Nothing else.
    
    Description: {description}
    """)
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"description": description})
    return "YES" in result.upper()

def is_industrial_sealing_product(description):
    # If it looks like a SKU code, skip the industrial check
    if "MS" in description.upper().replace(" ", "").replace("-", ""):
        return True
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    prompt = ChatPromptTemplate.from_template("""
    Is this product description related to industrial sealing, insulating, gasket, packing, or textile materials?
    Answer only YES or NO. Nothing else.
    
    Description: {description}
    """)
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"description": description})
    return "YES" in result.upper()

def map_sku(description):
    # Step 1 - Check if industrial product
    if not is_industrial_sealing_product(description):
        return "SKU NOT FOUND — Manual Review Required"
    
    # Step 2 - Match to SKU
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    prompt = ChatPromptTemplate.from_template("""
    Match this customer description to Mineral Seal Corporation's SKU codes.
    
    Customer description: {description}
    
    Available SKUs:
    MS-GR-050: Graphite rope packing (graphite cord, graphite rope, flexible graphite)
    MS-CF-T1: Ceramic fiber tape 1 inch
    MS-CF-T2: Ceramic fiber tape 2 inch
    MS-FG-CL1: Fiberglass cloth (glass fabric, woven glass, fiberglass sheet)
    MS-CF-CL1: Ceramic fiber cloth
    MS-BP-001: Braided packing
    MS-BP-002: Braided carbon packing (carbon fiber seal, carbon braided)
    MS-GR-075: Graphite rope packing 3/4 inch

    RULES:
    1. If description IS already a valid SKU code like MS-GR-050 return in format: MS-GR-050: Graphite rope packing
    2. If description contains a SKU code with spaces or different formatting like MS GR 050 or ms-gr-050 or MS - GR - 050 normalize it and return in format: MS-GR-050: Graphite rope packing
    3. If description matches a product name or vague description return the closest match in format: MS-GR-050: Graphite rope packing
    4. STRICT SIMILARITY CHECK: Before matching ask yourself - is this description referring to the EXACT SAME product category?
       - Graphite rope packing = rope or cord made of graphite for sealing
       - Ceramic fiber tape = tape made of ceramic fiber for insulation  
       - Fiberglass cloth = woven cloth made of fiberglass
       - Braided packing = braided fiber packing material
       - Rubber gasket is a DIFFERENT product category from all above — return NOT_FOUND
       - Only match if you are highly confident the products serve the same function and are made of similar materials.
       - When in doubt return NOT_FOUND
    5. Return ONLY the SKU code and official name in the format shown. No extra words or sentences.
    """)
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"description": description})
    
    if "NOT_FOUND" in result:
        return "SKU NOT FOUND — Manual Review Required"
    return result.strip()

def validate_po(extracted_text, customer_name):
    issues = []
    known_customers = ["ABC Manufacturing", "XYZ Industrial Supply"]
    if not any(c.lower() in customer_name.lower() for c in known_customers):
        issues.append("Customer not found in system — please verify")
    if "Net 30" not in extracted_text and "Net 60" not in extracted_text:
        issues.append("Payment terms need verification")
    return issues

# Streamlit UI
st.title("Mineral Seal PO Processing System")
st.caption("AI powered purchase order extraction and validation")

st.subheader("Upload a Purchase Order")
upload_option = st.radio("Choose input method:", ["Upload PDF", "Use Sample PO"])

po_text = None

if upload_option == "Upload PDF":
    uploaded_file = st.file_uploader("Upload PO (PDF)", type=["pdf"])
    if uploaded_file is not None:
        po_text = extract_text_from_pdf(uploaded_file)
        st.success("PDF uploaded successfully!")
else:
    po_text = load_po("sample_po.txt")
    st.info("Using sample PO for demonstration")

if po_text and st.button("Process PO"):
    with st.spinner("AI is processing your purchase order..."):

        st.subheader("Raw PO Content")
        st.text(po_text)

        st.subheader("AI Extracted Information")
        extracted = extract_po_info(po_text)
        st.write(extracted)

        st.subheader("SKU Mapping")
        line_items = extract_line_items(po_text)
        all_found = True
        for item in line_items:
            sku = map_sku(item)
            if "NOT FOUND" in sku:
                st.error(f"{item} → {sku}")
                all_found = False
            else:
                st.success(f"{item} → {sku}")

        st.subheader("Validation Results")
        customer_name = ""
        for line in extracted.split('\n'):
            if "Customer Name" in line:
                customer_name = line.replace("Customer Name:", "").strip()
                break

        issues = validate_po(extracted, customer_name)
        if issues:
            for issue in issues:
                st.warning(f"Issue: {issue}")
        else:
            st.success("All validation checks passed!")

        st.subheader("Ready for QuickBooks")
        if all_found and not issues:
            st.success("All checks passed! Ready to send to QuickBooks.")
            if st.button("Approve and Send to QuickBooks"):
                st.success("Sales Order sent to QuickBooks successfully!")
        else:
            st.warning("Please resolve flagged issues before sending to QuickBooks.")