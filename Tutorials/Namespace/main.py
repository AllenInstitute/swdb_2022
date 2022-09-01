import my_module


var = 20

print()
print("The namespace of the main script")
print("\tGlobals (keys of 'globals()')")
for key in globals().copy():
    print("\t\t"+key)

print()
print("\tLocals (keys of 'locals()')")
for key in locals():
    print("\t\t"+key)

print()
print("\tnames in the current local scope (output of 'dir()')")
for name in dir():
    print("\t\t"+name)


my_module.f(2)

print()
print("Accessing the name 'x' from 'my_module' from 'main' with code:")
print("print(my_module.x)\n\nwith output:")
print(my_module.x)
