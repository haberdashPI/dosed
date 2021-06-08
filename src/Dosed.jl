module Dosed

macro usepython()
    try
        ENV["PYCALL_JL_RUNTIME_PYTHON"] =
            strip(read(setenv(`poetry env info --path`, dir=homedir), String))
        using PyCall
    catch
        ENV["PYTHON"] =
            strip(read(setenv(`poetry env info --path`, dir=homedir), String))
        import Pkg; Pkg.build("PyCall")
        using PyCall
    end
end

end
