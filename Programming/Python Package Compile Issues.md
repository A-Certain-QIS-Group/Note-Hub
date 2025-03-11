# Python Package Compile Issues

### build python wheels using mingw (instead of MSVC)
*Sep.2 2024*
- locate/create C:\Python39\Lib\distutils\distutils.cfg
```ini
[build]
compiler = mingw32
```

### show package source when build error
```bash
pip download package_name --no-binary :all:
```