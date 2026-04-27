call .venv\scripts\activate

set DISTUTILS_USE_SDK=1

cd temp_consistency_module\networks\resample2d_package
python setup.py install
cd ..\..\..

cd temp_consistency_module\networks\channelnorm_package
python setup.py install
cd ..\..\..

cd temp_consistency_module\networks\correlation_package
python setup.py install
cd ..\..\..
