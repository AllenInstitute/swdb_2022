
print("The code in a 'module' file gets executed when it is imported.\nThat's where this print statement comes from.")

print()
print("The following code gets run in 'my_module'")
print("s = 'hello'\nx = 10")
s = "hello"
x = 10
print()
print("This defines 'x' and 's' in the namespace of 'my_module'")
print()


def f(y):
    print()
    print("Calling 'f' from 'my_module'.")
    print("The code inside 'f' is:")
    print()
    print("x = 20\nprint(x, y, s)\n\nwith output:")
    print()
    x = 20
    print(x, y, s)
    print()
    print()
    print("'s' is not local to 'f' but is in the namespace in which 'f' was defined and so is available.")
    print("'y' is the argument passed to 'f', and thus is local to 'f'.")
    print("'x' is a name defined in the local scope of 'f', and thus covers the 'x' in the namespace of 'my_module'")
    print()
    print("The namespace of the function 'f' in 'my_module'")
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
    
    


print("The namespace of my_module")
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



