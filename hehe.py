import easyocr

# Create an OCR reader object
reader = easyocr.Reader(['en'])

# Read text from an image
result = reader.readtext('image.jpeg')

# Combine extracted text into a single paragraph
paragraph = " ".join([detection[1] for detection in result])

# Print the paragraph
print(paragraph)
