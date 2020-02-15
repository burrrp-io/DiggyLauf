from fastapi import FastAPI, Query, Path, Body
from starlette.responses import HTMLResponse, FileResponse, StreamingResponse
import tempfile
import sqlite3
from sqlite3 import Error
import random
from PIL import Image
import torch
import torchvision
from ai.model import SimpleNet

app = FastAPI()

with open("../cats.txt", "r") as f:
    cats = f.read().split("\n")

cats_path = "/home/jannis/data/datasets/images/dogs-vs-cats/root/train/"

model = SimpleNet()
model.load_state_dict(torch.load("../sweet_cats.pth"))
model.eval()

create_table_sql = """ CREATE TABLE IF NOT EXISTS cats (
                                    id integer PRIMARY KEY AUTOINCREMENT,
                                    cat_id text NOT NULL,
                                    cute boolean
                                ); """
 
""" create a database connection to a SQLite database """
with sqlite3.connect("/home/jannis/sqlite3/pythonsqlite.db") as conn:
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
    with open("../index.html", "r") as f:
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
            path = "../src/"
            
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
        return FileResponse("../src/" + filename, media_type="text/css")
    elif ending == "js":
        return FileResponse("../src/" + filename, media_type="application/javascript")
    elif ending == "svg":
        return FileResponse("../src/" + filename, media_type="image/svg+xml")



    return 404

@app.get("/cat")
async def rate_cat(
    cat_url: str = Query(...),
    cute: bool = Query(...)
    ):
    cat_id = cat_url.split("/")[-1]
    task = (cat_id, cute)
    sql = ''' INSERT INTO cats(cat_id,cute)
              VALUES(?,?);'''
    print(sql)
    with sqlite3.connect("/home/jannis/sqlite3/pythonsqlite.db") as conn:
        cur = conn.cursor()
        cur.execute(sql, task)
        print(cur.lastrowid)

    # if cur.lastrowid%64 == 0:


    found = False
    while found == False:
        cat = cats[random.randint(0, len(cats))]
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
    
    return {"id": cat, "url": "src/" + cat}
