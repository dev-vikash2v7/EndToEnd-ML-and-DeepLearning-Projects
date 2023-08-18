function onPageLoad() {
    const Http = new XMLHttpRequest();
    var url = "http://127.0.0.1:5000/get_location_name"

    Http.open("GET", url);
    Http.send();

    Http.onreadystatechange = (e) => { // data  = eval(e.target.responseText)
        if (e.target.responseText) {
            data = JSON.parse(e.target.responseText)

            var locations = data.locations;
            var uiLocaions = document.getElementById("uiLocations")
            // uiLocaions.empty()

            
            for (var i in locations) {
                var opt = new Option(locations[i])
                if(i==0) {
                        opt.selected = true ;
                }
                opt.className = "uiLocation" ;
                uiLocaions.append(opt)
            }
        }
    }


    // $.get(url, function (data, status) {

    //     console.log("get location names")
    //     console.log(data)
    // if (data) {
    //     var locations = data.locations;
    //     var uiLocaions = document.getElementById("uiLocations")
    //     uiLocaions.empty()

    //     for (var location in locations) {
    //         var opt = new Option(location)
    //         uiLocaions.append(opt)
    //     }
    // }
}

function getValue(name) {
    uiBox = document.getElementsByName(name);

    for (var i in uiBox) 
        if (uiBox[i].checked) {
            return parseInt(uiBox[i].value) // parseInt(i) + 1;
        }
}


function onClickPricePrediction() {
    console.log("price btn clickd")

    var total_sqft = parseInt(document.getElementById("uiSqft").value)
    var bhk = getValue("uiBhk")
    var bath = getValue("uiBath")

    var locations = document.getElementsByClassName("uiLocation");
    var location;
    for (var i in locations) {
        if (locations[i].selected) {
            location = locations[i].innerText
            break;
        }
    }

    var pred_price = document.getElementById("predicted_price")

    var url = "http://127.0.0.1:5000/predict_price"

    const xhr = new XMLHttpRequest();

    xhr.open("POST", url);
    data = {
        total_sqft,
        bhk,
        bath,
        location
    };
    data = JSON.stringify(data);

    fetch(url, {
        method: "POST",
        mode: 'cors',
        credentials: "include",
        body: data,

        headers: {
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
    }).then(response => response.json()).then(data => pred_price.innerText = data["predicted_price"] + " RS").catch(err => console.log(`errrrrrrrr ${err} `))
}
window.onload = onPageLoad
