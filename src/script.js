const testcats = ["src/cat.2.jpg", "src/cat.1.jpg", "src/cat.3.jpg", "src/cat.0.jpg"] //{"id":"cat.3456.jpg"}
var i = 0

function vote(id){
    document.getElementById(id).blur();

    if(id === "sweet"){
        console.log("sweet")
        
    }else{
        console.log("ugly ugly cat")
    }

    if (i < testcats.length) {
        document.getElementById("cat").setAttribute("src", testcats[i]);
        i= i+1;
        console.log("variable i ", i);
    } else{
        document.getElementById("cat").setAttribute("src", testcats[2]);
    }
    

}
