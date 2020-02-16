from fastapi import FastAPI, Query, Path, Body, BackgroundTasks
from starlette.responses import HTMLResponse, FileResponse, StreamingResponse
import tempfile
import sqlite3
from sqlite3 import Error
import random
from PIL import Image
import torch
import torchvision
from app.ai.model import SimpleNet
from app.ai.training import learn

app = FastAPI()

with open("/app/cats.txt", "r") as f:
    cats = f.read().split("\n")

cats_path = "/app/data/root/train/"

model = SimpleNet()
model.load_state_dict(torch.load("/app/sweet_cats.pth"))
model.eval()

create_table_sql = """ CREATE TABLE IF NOT EXISTS cats (
                                    id integer PRIMARY KEY AUTOINCREMENT,
                                    cat_id text NOT NULL,
                                    cute boolean
                                ); """
 
""" create a database connection to a SQLite database """
with sqlite3.connect("/app/sqlite3/pythonsqlite.db") as conn:
    print(sqlite3.version)
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
    except Error as e:
        print(e)


@app.get("/")
# function name is used as description in swagger UI
# # optional query parameters require a "None" default value
async def get_query():
    with open("/app/index.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.get("/src/{filename}")
def get_src(
    filename: str = Path(...)
    ):
    print("it should work here", filename)
    ending = filename.split(".")[-1]
    print("over",ending)
    if ending == "png" or ending == "jpg" or ending == "ico":
        if filename[0:3] == "cat":
            path = cats_path
        else:
            path = "/app/src/"

        with open(path + filename, "rb") as f:
            img = f.read()
            byte_img = bytearray(img)
            with tempfile.NamedTemporaryFile(mode="w+b", suffix="." + ending, delete=False) as FOUT:
                FOUT.write(byte_img)
                if ending == "png":
                    return FileResponse(FOUT.name, media_type="image/png")
                elif ending == "jpg":
                    return FileResponse(FOUT.name, media_type="image/jpg")
                elif ending == "ico":
                    return FileResponse(FOUT.name, media_type="image/x-icon")
    
    if  ending == "css":
        return FileResponse("/app/src/" + filename, media_type="text/css")
    elif ending == "js":
        return FileResponse("/app/src/" + filename, media_type="application/javascript")
    elif ending == "svg":
        return FileResponse("/app/src/" + filename, media_type="image/svg+xml")



    return 404

@app.get("/cat")
async def rate_cat(
    cat_url: str = Query(...),
    cute: bool = Query(...)
    ):

    global model

    db = "/app/sqlite3/pythonsqlite.db"
    train_it = False
    cat_id = cat_url.split("/")[-1]
    task = (cat_id, cute)
    sql = ''' INSERT INTO cats(cat_id,cute)
              VALUES(?,?);'''
    print(sql)
    with sqlite3.connect(db) as conn:
        cur = conn.cursor()
        cur.execute(sql, task)
        print("last row", cur.lastrowid, flush=True)
        last = cur.lastrowid
        if last%64 == 0:
            train_it = True


    found = False
    while found == False:
        cat = cats[random.randint(0, len(cats))]
        print("cat",cat,flush=True)
        pic = Image.open(cats_path+cat)
        transform = torchvision.transforms.Compose([
                    torchvision.transforms.Resize((32,32)),
                    torchvision.transforms.ToTensor()
                ])
        scaled = transform(pic)
        unsqueezed = scaled.unsqueeze(0)
        result = model(unsqueezed)
        label = torch.max(result, 1)[1][0].item()
        if label == 0:
            found=True

    if train_it == True:
        with sqlite3.connect(db) as conn:
            cur = conn.cursor()
            cur.execute("SELECT cat_id, cute FROM cats WHERE id>?", (last-64,))
            liste = cur.fetchall()
            model = learn(liste, model)
            print("successfully trained", flush=True)


    
    return {"id": cat, "url": "src/" + cat}
