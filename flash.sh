#!/bin/bash
#Script to flash buildroot image to micro-sd card
#Author: Brad Waggle

IMAGE_TO_FLASH="buildroot/output/images/sdcard.img"
MICRO_SD_DEVICE="/dev/sdb"

sudo dd if=$IMAGE_TO_FLASH of=$MICRO_SD_DEVICE conv=fdatasync bs=1M