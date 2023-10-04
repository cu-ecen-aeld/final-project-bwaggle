OPENCV_PYTHON_VERSION = 4.8.1.78
OPENCV_PYTHON_SITE = https://files.pythonhosted.org/packages/c0/52/9fe76a56e01078a612812b40764a7b138f528b503f7653996c6cfadfa8ec
OPENCV_PYTHON_SOURCE = opencv-python-$(OPENCV_PYTHON_VERSION).tar.gz
#OPENCV_PYTHON_SOURCE = https://files.pythonhosted.org/packages/c0/52/9fe76a56e01078a612812b40764a7b138f528b503f7653996c6cfadfa8ec/opencv-python-4.8.1.78.tar.gz
# OPENCV_PYTHON_SOURCE = opencv-python-4.8.1.78.tar.gz
OPENCV_PYTHON_LICENSE = MIT
OPENCV_PYTHON_LICENSE_FILES = LICENSE.txt
OPENCV_PYTHON_SETUP_TYPE = setuptools

# define OPENCV_PYTHON_BUILD_CMDS
#     $(MAKE) -C $(@D) all
# endef

# define OPENCV_PYTHON_INSTALL_TARGET_CMDS
#     $(INSTALL) -D -m 755 $(@D)/build/lib/python3/cv2.so $(TARGET_DIR)/usr/lib/python3.10/site-packages/cv2.so
# endef

$(eval $(generic-package))
