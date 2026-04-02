# Nightreign Recognizer - User Guide

## What this tool does

This tool captures the Nightreign map from your screen, identifies which map variation you are currently viewing, and then shows a completed reference image with full map information.

The application is designed to work while the game remains in focus. It reads the map from the screen, briefly shows the cropped screenshot that it used for recognition, and then replaces that preview with a completed detailed map image.

---

## Before you start recognition

Before triggering recognition, make sure all of the following are true:

- the **map is open** in the game,
- the map is **visible on the selected screen**,
- the map is **not covered** by other windows or overlays,
- the map is **not filtered or altered** by visual effects that change its appearance.

Examples of things that can interfere with recognition:
- overlays sitting on top of the map,
- color filters,
- streaming or video-player overlays,
- partially hidden map UI,
- post-processing that changes how the map looks,
- selecting the wrong monitor in the tool.

---

## Important timing note

The map only needs to remain open **while the tool is taking the screenshot**.

You can tell that the screenshot step is complete when you see the **cropped screenshot of the map** appear in the main image area of the program.

That cropped screenshot appears **before** the final completed map image appears.

So the normal sequence is:
1. you start recognition,
2. the tool captures the screen and crops the map,
3. the cropped map screenshot appears in the app,
4. the tool finishes recognition,
5. the completed detailed map image appears afterward.

Once the cropped screenshot has appeared, the capture step has already happened for that run.

---

## Basic usage

### 1. Open the program
Launch the application.

### 2. Choose the correct screen
If you use more than one monitor, select the display that contains the game map.

If your game is on the main monitor, you can usually leave the selection on **Primary / default**.

### 3. Open the map in the game
Make sure the in-game map is open and clearly visible.

### 4. Start recognition
Start recognition either by:
- pressing the main recognition button, or
- using the configured global hotkey if enabled.

### 5. Wait for the cropped map screenshot
The first image you will usually see is the **cropped screenshot of the map** that the tool is actually using for recognition.

This is an intermediate step and is expected.

### 6. Wait for the final completed map image
After recognition finishes, the program replaces the cropped screenshot with the completed reference map image that contains the full details.

### 7. Review the alternatives
The right side of the app shows the top alternative matches.

You can click them to replace the main image if you want to inspect another candidate.

---

## What the tool is doing internally

In simple terms, the tool does the following:

1. captures the selected display,
2. finds the map area on the screen,
3. crops the map from the screenshot,
4. analyzes the map,
5. compares the recognized landmark pattern against its packaged recognition data,
6. ranks the most likely map variations,
7. displays the best result and top alternatives.

This is why you first see the cropped screenshot and only later see the completed detailed map image.

---

## Reading the result area

The main large image area shows:
- first: the cropped map screenshot used for recognition,
- later: the best completed map image.

The right-hand column shows several top candidate maps.

Clicking one of those candidates replaces the large image with that candidate.

---

## Zoom and drag controls

The main large image supports:
- zoom in / zoom out,
- click-and-drag panning,
- reset-to-fit by double-clicking.

This is useful when you want to inspect details closely.

---

## Hotkey support

The application can come with a default global hotkey already configured.

You can also change the hotkey from inside the program through the settings gear icon.

Typical hotkey workflow:
1. click the gear icon,
2. open the hotkey settings popover,
3. click the button to set a new hotkey,
4. press the combination you want,
5. save or apply it.

If a hotkey is already in use by another program, Windows may reject it.

---

## Tips for best results

For more reliable recognition:
- keep the map fully visible,
- avoid overlays over the map,
- make sure the correct screen is selected,
- let the app finish the run before judging the result,
- use the top alternatives when the best result looks suspicious.

---

## If recognition looks wrong

Check these first:
- Was the game map fully open?
- Was the correct monitor selected?
- Was the map covered by overlays?
- Did the cropped screenshot look correct?

If the cropped screenshot itself looks wrong, the issue likely happened before recognition.

If the cropped screenshot looks correct but the final map is wrong, the issue likely happened during recognition.

---

## Files used by the application

Depending on the build, the app may rely on:
- a packaged recognition data file,
- packaged completed map images,
- a settings file created automatically by the application.

These are normally handled for you in the packaged product.

---

## Product note

This README is written so it can be shipped with a packaged executable and remain suitable for future commercial distribution.

If this tool is distributed commercially, the exact feature set, bundled data, and support policies may differ from a development build.

---

## Quick-start summary

1. Launch the app.
2. Select the correct screen.
3. Open the in-game map and keep it visible.
4. Start recognition.
5. Wait until the cropped map screenshot appears.
6. Wait a little longer for the completed detailed map image.
7. Review or click the alternatives if needed.
