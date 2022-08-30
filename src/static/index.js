// refresh board svg image every second (1000 milliseconds)
const img = document.querySelector("img");
setInterval(function(){ img.src = "../static/board.svg?" + Math.random();}, 1000)


// display/hide "Play move" textfield depending on chosen game mode
const el = document.getElementById("gamemode");

const playerMoveDiv = document.getElementById("playerMoveDiv");

el.addEventListener("change", function handleChange(event) 
{
if (event.target.value === "player") playerMoveDiv.style.display = "block";
else playerMoveDiv.style.display = "none";
});


// disable/enable some buttons according to click of other buttons
const gameModeForm = document.getElementById("gameModeForm");
const resetGameForm = document.getElementById("resetGameForm");
const playMoveForm = document.getElementById("playMoveForm");
const startGameButton = document.getElementById("startGameButton");
const resetGameButton = document.getElementById("resetGameButton");
const playMoveButton = document.getElementById("playMoveButton");

gameModeForm.onsubmit = function()
{
    startGameButton.disabled = true;
    resetGameButton.disabled = false;
    playMoveButton.disabled = false;
}

resetGameForm.onsubmit = function() 
{
    startGameButton.disabled = false;
    resetGameButton.disabled = true;
    playMoveButton.disabled = true;
    document.getElementById("enteredMove").value = "";
}

// reset the "Play move" textfield after a move has been entered
playMoveForm.onsubmit = function()
{
    playMoveForm.submit();
    document.getElementById("enteredMove").value = "";
    return false;
}

