#!/usr/bin/env python3
"""Extract text from RemoteSAM.pdf"""
import PyPDF2

with open('RemoteSAM.pdf', 'rb') as file:
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n\n"
    
with open('RemoteSAM_extracted.txt', 'w', encoding='utf-8') as output:
    output.write(text)

print(f"Extracted {len(reader.pages)} pages")
print(f"Total characters: {len(text)}")
