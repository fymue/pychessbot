// refresh game board svg image and move history every second (1000 milliseconds)

const moveHistoryFile = "../static/move_history.txt";
const iframe = document.getElementById("moveHistory");
const img = document.querySelector("img");

async function updateGame()
{
    img.src = "../static/board.svg?" + Math.random();
    iframe.contentWindow.document.open();
    await fetch(moveHistoryFile)
          .then(response => response.text())
          .then(data => iframe.contentWindow.document.write(data));
    iframe.contentWindow.document.close();
}

setInterval(updateGame, 1000)

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

