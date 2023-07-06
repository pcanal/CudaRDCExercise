#include <dlfcn.h>
#include <iostream>

// void foo();
using fooptr_t = void (*)();

int main()
{
    // `mod_foo` is not a MODULE so we need to explicitly load the final
    // library
    void *handle = dlopen("./libmod_foo_final.so", RTLD_LAZY | RTLD_GLOBAL);
    if (!handle)
    {
        std::cerr << "Error: could not open library libmod_foo.so\n";
        return 1;
    }
    void *voidfunc = dlsym(handle, "_Z3foov");
    if (!voidfunc)
    {
        std::cerr << "Error: could not find the symbol for the foo function\n";
        return 2;
    }
    fooptr_t foo = reinterpret_cast<fooptr_t>(voidfunc);
    foo();

    return dlclose(handle);
}