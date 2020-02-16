FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

WORKDIR /usr
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install aiofiles==0.4.0

COPY cats.txt /app/cats.txt
COPY sweet_cats.pth /app/sweet_cats.pth
RUN mkdir /app/sqlite3
RUN touch /app/sqlite3/pythonsqlite.db
RUN mkdir /app/data
RUN mkdir /app/data/root
RUN mkdir /app/data/root/train
RUN mkdir /app/src
COPY ./src/ /app/src/
COPY ./index.html /app/index.html

COPY ./app /app/app
