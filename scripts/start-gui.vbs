Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

root = fso.GetParentFolderName(fso.GetParentFolderName(WScript.ScriptFullName))
exe = root & "\target\release\bidking-analyzer.exe"

shell.CurrentDirectory = root
shell.Run """" & exe & """", 1, False
