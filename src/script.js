const testcats = ["src/cat.2.jpg", "src/cat.1.jpg", "src/cat.3.jpg", "src/cat.0.jpg"] //{"id":"cat.3456.jpg"}
var i = 0

function init() {
    var rand = 0;
    while(rand == 0) {
        rand = Math.random() * 25000;
    }

    setSource(`src/cat.${Math.round(rand)}.jpg`)
}

function vote(id){
    document.getElementById(id).blur();
    var cat_url= document.getElementById("cat").getAttribute("src");
    var xmlHttp = new XMLHttpRequest();
    xmlHttp.open("GET", `/cat?cat_url=${cat_url}&cute=${(id=== "sweet")}`, false);
    xmlHttp.send();

    console.log("response", xmlHttp.responseText)
    var newcat = JSON.parse(xmlHttp.responseText)

    if(id === "sweet"){
        console.log("sweet")
        
    }else{
        console.log("ugly ugly cat")
    }
    setSource(newcat.url)

}

function setSource(source){
    document.getElementById("cat").setAttribute("src", source);  
}
