#!/bin/bash
#Script to flash buildroot image to micro-sd card
#Author: Brad Waggle

IMAGE_TO_FLASH="buildroot/output/images/sdcard.img"
MICRO_SD_DEVICE="/dev/sdc"

sudo umount /dev/sdc1
sudo umount /dev/sdc2


sudo dd if=$IMAGE_TO_FLASH of=$MICRO_SD_DEVICE conv=fdatasync bs=1M
udisksctl power-off -b /dev/sdc
