FROM public.ecr.aws/lambda/python:3.10

# Copy requirements.txt
COPY requirements.txt .
COPY insightface-0.7.3-cp310-cp310-win_amd64.whl .
#RUN pip install insightface-0.7.3-cp310-cp310-win_amd64.whl --target "${LAMBDA_TASK_ROOT}"
RUN pip install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

# Adjusted the path to copy the 'src' directory

# Copy function code
COPY app.py ${LAMBDA_TASK_ROOT}

COPY ./* ./AVATAR/

COPY __init__.py .
COPY inswapper_128.onnx .
COPY .insightface ./.insightface


CMD [ "app.handler" ]
