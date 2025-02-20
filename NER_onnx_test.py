from NER_onnx import NER_Processing

NER = NER_Processing()
text = "George Washington was born on February 22, 1732 at Popes Creek in Westmoreland County, Virginia. He was the first of six children of Augustine and Mary Ball Washington."
output = NER.predict(text)
print(output)