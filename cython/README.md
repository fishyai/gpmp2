Building GPMP2 Python Bindings
===================================================
This is a mostly-untested, experimental Cython wrapper around GPMP2. It uses the
GTSAM Python wrappers.
Prerequisites
------

- Build the GTSAM Cython bindings according to the instructions [here](https://github.com/borglab/gtsam/tree/develop/cython). Be sure to update your PYTHONPATH variable.


Installing the Toolbox
-----

```
$ cmake -DGPMP2_BUILD_CYTHON_TOOLBOX:OPTION=ON -DGTSAM_TOOLBOX_INSTALL_PATH:PATH=/path/install/toolbox ..
$ make install
```

Note that the GTSAM_TOOLBOX_INSTALL_PATH will automatically set to
/usr/local/gtsam_toolbox if you do not set a value

