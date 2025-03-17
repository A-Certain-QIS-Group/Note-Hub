# Linux and Windows Commands


## Screen and Process Management

Screens can help you run time consuming tasks in background.

you can run your task in a "Screen". The Screen is like a shell (command window) but keep opened all the time. When you log out, the tasks in the Screen will remain running. You can Attach to the same screen when you come back

### Create, Leave and Reenter Screens
- Create a new Screen
    ```bash
    screen -S your_screen_name
    ```
- Leave the current screen
    `Ctrl+A`, then `D`
- Attach to existing screen
    ```bash
    screen -rd your_screen_name
    ```

hint: try to not enter another screen inside a screen.

### Delete unwanted screen

```bash
screen -rd screen_to_delete # enter your screen
```
then `Ctrl+A`, then `K`

In case you accidentally created two screens with the same name, you can list all the screens and kill the unwanted one

```bash
screen -ls
    # This is for listing all your screens. 
    # The format is screen_id.screen_name. Screens with different names have the same id
screen -X -S your_screen_id kill
```

### Kill Unused Screen
```bash
screen -r name
Ctrl + A, then K
```
or
```bash
screen -ls
screen -X -S <session_id> quit
```

### Find and Kill a Process
```bash
ps aux | grep run_find_critical_v20e_scan_JII1.py
kill <PID>
```
force kill
```
kill -9 <PID>
```

### Pause a Process

```bash
kill -STOP <PID> 
    # pause it (the word "Stop" is misleading)
kill -CONT <PID> 
    # resume it
```

### Run Multiple Processes Sequentially

```bash
julia folder/script1.jl ; julia folder/script2.jl
```

## Symbolink

### Create Symbolink
```bash
ln -s /target/folder /link/entrance
```
dont put "/" at the end of the entrance folder name

#### For windows: 
```powershell
New-Item -Path "C:\Path\Your\Link\Appears\LinkName" -ItemType SymbolicLink -Target "C:\Path\Linking\To"
```

however, it dont work well with onedrive sync. if you want to put your code on onedrive and data in another place, unfortunately, onedrive will copy all the symbolically linked folders

```cmd
mklink /D link_entrance "D:\TARGET_FOLDER"
```

TODO check if onedrive will also sync this

### Remove the link without deleting the original object
```bash
rm ./link/entrance
```

## Misc

### Decompress a file in Linux
```bash
tar -xvzf file.tar.gz # decompress
tar -cvzf file.tar.gz folder # compress
tar -cvf - folder | pigz -p 8 > file.tar.gz # compress with 8 cores
```
-x: Extract files.
-v: Verbose mode (optional; shows the extraction process).
-z: Uncompress the file using gzip.
-f: Specifies the filename to extract.
-: output to stdout
