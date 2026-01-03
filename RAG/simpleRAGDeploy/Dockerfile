# Use the AWS base image for python 3.12
FROM public.ecr.aws/lambda/python:3.12

# Install buid-essential compiler and tool
RUN microdnf update -y && \
    microdnf install -y gcc-c++ make

# Copy requirements file
COPY requirements.txt ${LAMBDA_TASK_ROOT}

# Install dependencies
RUN pip install -r requirements.txt

#Copy PDF file
COPY DOC-SF238339076816-20230503.pdf ${LAMBDA_TASK_ROOT}/DOC-SF238339076816-20230503.pdf

# Copy the lambda function code
COPY simplerag.py ${LAMBDA_TASK_ROOT}

# Set permission to execute the file
RUN chmod +x simplerag.py

# Set the CMD to your handler 
CMD ["simplerag.lambda_handler"]