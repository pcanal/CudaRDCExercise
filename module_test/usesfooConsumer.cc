#include <dlfcn.h>
#include <iostream>

// void foo();
using fooptr_t = void (*)();

int main()
{
    void *handle = dlopen("./libuses_foo.so", RTLD_LAZY | RTLD_GLOBAL);
    if (!handle)
    {
        std::cerr << "Error: could not open library libuses_foo.so\n";
        return 1;
    }
    void *voidfunc = dlsym(handle, "_Z7usesfoov");
    if (!voidfunc)
    {
        std::cerr << "Error: could not find the symbol for the usesfoo function\n";
        return 2;
    }
    fooptr_t usesfoo = reinterpret_cast<fooptr_t>(voidfunc);
    usesfoo();

    return dlclose(handle);
}