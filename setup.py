from setuptools import setup

package_name = 'local_learning'

setup(
    name=package_name,
    version='0.0.0',
    package_dir={'': 'src'},
    py_modules=['local_learning_node'],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'params.yaml']),
        ('share/' + package_name + '/model', ['model/mobilenet_trained_updated.pt']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Maintainer',
    maintainer_email='example@example.com',
    description='Local planner using a LibTorch model.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'local_learning_node = local_learning_node:main',
        ],
    },
)
