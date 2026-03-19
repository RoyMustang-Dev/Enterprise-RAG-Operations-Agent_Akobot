with open('app/main.py', 'r', encoding='utf-8') as f:
    text = f.read()

target_import = 'from app.api.routes import router as api_router'
new_import = target_import + '\nfrom app.v2.api.routes_v2 import router as api_router_v2'
text = text.replace(target_import, new_import)

target_mount = 'app.include_router(api_router, prefix="/api/v1")'
new_mount = target_mount + '\napp.include_router(api_router_v2, prefix="/api/v2")'
text = text.replace(target_mount, new_mount)

with open('app/main.py', 'w', encoding='utf-8') as f:
    f.write(text)
print('Patched main.py')
