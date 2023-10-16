#!/bin/bash
#Script to flash buildroot image to micro-sd card
#Author: Brad Waggle

# IMAGE_TO_FLASH="buildroot/output/images/sdcard.img"
# MICRO_SD_DEVICE="/dev/sdb"

# sudo umount $MICRO_SD_DEVICE
# sudo umount $MICRO_SD_DEVICE


# sudo dd if=$IMAGE_TO_FLASH of=$MICRO_SD_DEVICE conv=fdatasync bs=1M
# udisksctl power-off -b $MICRO_SD_DEVICE

#!/bin/bash

# Define the directory location of the image file
image_dir="buildroot/output/images"
image_file="sdcard.img"


# List block devices and filter for removable drives, excluding /dev/sda
devices=$(lsblk -l -o NAME,TYPE,SIZE,TRAN | grep -E 'disk|usb' | awk '{print $1}' | grep -v 'sda')

# Check if there are no removable drives
if [ -z "$devices" ]; then
  echo "No removable drives found. Please insert the micro SD card."
  exit 1
fi

# Prompt the user to select the device
PS3="Select the micro SD card device: "
select device in $devices; do
  if [ -n "$device" ]; then
    device="/dev/$device"
    echo "You selected $device."
    break
  else
    echo "Invalid option. Please choose a valid device."
  fi
done

# Confirm with the user before proceeding
read -p "Are you sure you want to write the image to $device? (y/n): " confirm
if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
  echo "Operation aborted."
  exit 0
fi

# Check if the image file exists
if [ ! -f "$image_dir/$image_file" ]; then
  echo "Image file not found in $image_dir. Please specify the correct location and file name."
  exit 1
fi

# Find all mountpoints for the selected device
mountpoints=($(df -h | grep "$device" | awk '{print $6}'))

# Unmount all mountpoints associated with the device
for mountpoint in "${mountpoints[@]}"; do
    sudo umount "$mountpoint"
    echo "Device $device unmounted from $mountpoint."
done

# Use dd to write the image to the selected device
echo "Writing the image to $device. This may take some time. Please be patient."
sudo dd if="$image_dir/$image_file" of="$device" bs=4M status=progress

# Sync and eject the device
sync
echo "Image write completed. The micro SD card can now be safely removed."

# Power off the device
udisksctl power-off -b $device

