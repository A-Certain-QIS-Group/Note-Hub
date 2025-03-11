# Jupyter Notebook

### Using Jupyter lab on workstation
Do port forwarding (Remark: This also works when you are using screen)
ssh with port forwarding using 
```bash
ssh -L [local port]:localhost:[remote port] [username]@[domain name]
```
When launching jupyter, specify the port
```bash
jupyter lab --no-browser --port [remote port]
```
Copy the jupyter link to your browser, change the port number to `local port` 
