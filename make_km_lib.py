import requests
import io
import zipfile
import os

# download original code

km_url = "http://web.eecs.umich.edu/~cscott/code/kernel_MPE.zip"
km_filename = "Kernel_MPE_grad_threshold.py"

response = requests.get(km_url, allow_redirects=True)
km_code = None

with zipfile.ZipFile(io.BytesIO(response.content)) as thezip:
        for zipinfo in thezip.infolist():
            if zipinfo.filename == km_filename:
                km_code = thezip.open(zipinfo).read().decode(encoding='utf-8', errors='strict')
                
assert(km_code)


# Make it python3 compatible (very naively) and make sure that the sqrt of a negative number returns 0


replacements= [
    ('sqrt(distance_sqd)', 'sqrt(max(0,distance_sqd))'),
    ('print "kappa_star={}".format(kappa_star)', 'print ("kappa_star={}".format(kappa_star))'),
    ('print "KM1_estimate={}".format(KM1)','print ("KM1_estimate={}".format(KM1))'),
    ('print "KM2_estimate={}".format(KM2)','print ("KM2_estimate={}".format(KM2))'),
    ]

for orig,new in replacements:
    km_code = km_code.replace(orig,new)
    
os.makedirs("lib/km/km", exist_ok=True)

with open("lib/km/km/"+km_filename, "w+") as f:
    f.write(km_code)
    

setup = """import setuptools

setuptools.setup(
    name="km",
    description="Kernel Mixture Proportion Estimation",
    version="0.0.1",
    author="Ramaswamy",
    packages=setuptools.find_packages(),
)
"""


with open("lib/km/setup.py", "w+") as f:
    f.write(setup)
    


# add setup