## install
ubuntu终端执行
```bash
pip install -r requirements.txt
```

## 串口配置
ubuntu终端执行
```bash
# bash
sudo apt update
sudo apt install linux-tools-generic hwdata -y
sudo update-alternatives --install /usr/local/bin/usbip usbip \
    /usr/lib/linux-tools/$(ls /usr/lib/linux-tools)/usbip 20
```

windows终端查看
```powershell
# powershell
usbipd list
```

绑定设备
```powershell
# powershell
usbipd bind --busid 2-1 
usbipd attach --wsl --busid 2-1 # 2-1修改为实际的数传输出号
```
