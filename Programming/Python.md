# Python Programming Knowledge
**This is not a zero-to-hero textbook/guide! just cheatsheet for experienced programmer or newcompers to quick jump in**

## Multiprocessing
### Multiprocessing in jupyterlab without keep connection alive:
Using the multiprocessing package, feed functions that save output to files.
p = multiprocessing.Process(target=f,args=[args])
p.start()
