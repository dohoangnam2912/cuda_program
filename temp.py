import numpy as np

arr = [0.0011780261993408203, 0.0011725425720214844, 0.0011632442474365234, 0.001180887222290039, 0.0011751651763916016, 0.0011756420135498047, 0.0011632442474365234, 0.001180887222290039, 0.001161813735961914, 0.0011713504791259766, 0.0011582374572753906, 0.0011615753173828125, 0.0011734962463378906, 0.0011646747589111328, 0.0011723041534423828, 0.0011763572692871094, 0.0011820793151855469, 0.0011610984802246094, 0.0011754035949707031, 0.0012137889862060547, 0.0011754035949707031, 0.0011615753173828125, 0.0011806488037109375, 0.0011713504791259766, 0.0011670589447021484, 0.001163482666015625, 0.0011658668518066406, 0.0011703968048095703, 0.0011641979217529297, 0.0011696815490722656, 0.0011718273162841797, 0.001184701919555664, 0.0011708736419677734, 0.0011715888977050781, 0.0011599063873291016, 0.0011639595031738281, 0.0011625289916992188, 0.00116729736328125, 0.0011684894561767578, 0.0011794567108154297, 0.0011696815490722656, 0.0011570453643798828, 0.0011692047119140625, 0.0011570453643798828, 0.0011653900146484375, 0.0011696815490722656, 0.0011963844299316406, 0.0011708736419677734, 0.0011622905731201172, 0.0011637210845947266, 0.0011639595031738281, 0.001169443130493164, 0.0011594295501708984, 0.0011720657348632812, 0.0011742115020751953, 0.0011696815490722656, 0.0011646747589111328, 0.0011599063873291016, 0.0011577606201171875, 0.001161813735961914, 0.001157522201538086, 0.0011606216430664062, 0.0011658668518066406, 0.0011584758758544922, 0.0012271404266357422, 0.0012350082397460938, 0.0011603832244873047, 0.0012629032135009766, 0.0011589527130126953, 0.001161813735961914, 0.0011544227600097656, 0.0011675357818603516, 0.0011601448059082031, 0.0011692047119140625, 0.0011620521545410156, 0.0011608600616455078, 0.0011587142944335938, 0.0013108253479003906, 0.0012400150299072266, 0.001172780990600586, 0.0011692047119140625, 0.001169443130493164, 0.001155853271484375, 0.0011622905731201172, 0.0011599063873291016, 0.0011622905731201172, 0.0011594295501708984, 0.001171112060546875, 0.0011570453643798828, 0.0011878013610839844, 0.001402139663696289, 0.0011687278747558594, 0.0011630058288574219, 0.0011928081512451172, 0.001168966293334961, 0.0011758804321289062, 0.0011725425720214844, 0.0011622905731201172, 0.0011603832244873047, 0.00116729736328125, 0.0011603832244873047, 0.001157522201538086, 0.0011742115020751953, 0.0011582374572753906, 0.0011644363403320312, 0.0011591911315917969, 0.0011806488037109375, 0.0011706352233886719, 0.0011584758758544922, 0.0011744499206542969, 0.0011675357818603516, 0.0011761188507080078, 0.0011563301086425781, 0.0011761188507080078, 0.0011572837829589844, 0.0011625289916992188, 0.001171112060546875, 0.0011582374572753906, 0.0011610984802246094, 0.0011568069458007812, 0.0011627674102783203, 0.0011668205261230469, 0.0011713504791259766, 0.0011610984802246094, 0.001173257827758789, 0.0011649131774902344, 0.001161813735961914, 0.0011796951293945312, 0.0011594295501708984, 0.0011670589447021484, 0.0011603832244873047, 0.001157999038696289, 0.0011653900146484375, 0.0011584758758544922, 0.0011920928955078125, 0.0012118816375732422, 0.0011610984802246094, 0.0011584758758544922, 0.0011627674102783203, 0.0011570453643798828, 0.0011582374572753906, 0.001161813735961914, 0.0011560916900634766, 0.0011684894561767578, 0.0011551380157470703, 0.0011625289916992188, 0.001176595687866211, 0.0011644363403320312, 0.0011620521545410156, 0.0011661052703857422, 0.0011603832244873047, 0.0011653900146484375, 0.0011594295501708984, 0.0012581348419189453, 0.0011734962463378906, 0.0011606216430664062, 0.0011589527130126953, 0.0011813640594482422, 0.0011763572692871094, 0.0011644363403320312, 0.0011701583862304688, 0.0011620521545410156, 0.0011646747589111328, 0.0011622905731201172, 0.0011649131774902344, 0.001172780990600586, 0.0012896060943603516, 0.0013079643249511719, 0.0011763572692871094, 0.001165151596069336, 0.0011606216430664062, 0.00118255615234375, 0.001199483871459961, 0.00118255615234375, 0.0011687278747558594, 0.0011584758758544922, 0.0011713504791259766, 0.0011582374572753906, 0.0011539459228515625, 0.0011806488037109375, 0.0011949539184570312, 0.0011632442474365234, 0.0011777877807617188, 0.0011587142944335938, 0.001169443130493164, 0.0011608600616455078, 0.001150369644165039, 0.0011599063873291016, 0.001161336898803711, 0.0011837482452392578, 0.0011720657348632812, 0.0011556148529052734, 0.0011658668518066406, 0.0011682510375976562, 0.0011620521545410156, 0.0011670589447021484, 0.0011594295501708984, 0.0011587142944335938, 0.0011882781982421875, 0.0011546611785888672, 0.0011548995971679688, 0.0011551380157470703, 0.0011632442474365234, 0.0011591911315917969, 0.001148223876953125, 0.001154184341430664, 0.0011632442474365234, 0.0011565685272216797, 0.0011615753173828125, 0.0011539459228515625, 0.001150369644165039, 0.001184701919555664, 0.0011529922485351562, 0.0011539459228515625, 0.0011510848999023438, 0.0011522769927978516, 0.0011630058288574219, 0.0011548995971679688, 0.0011506080627441406, 0.0011587142944335938, 0.001157522201538086, 0.0011560916900634766, 0.0011484622955322266, 0.0011525154113769531, 0.0011637210845947266, 0.001157999038696289, 0.0011677742004394531, 0.0011546611785888672, 0.0011568069458007812, 0.0011701583862304688, 0.0011556148529052734, 0.0011565685272216797, 0.0011653900146484375, 0.00115966796875, 0.0011527538299560547, 0.0011553764343261719, 0.0011534690856933594, 0.0011594295501708984, 0.001191854476928711, 0.0011723041534423828, 0.001165628433227539, 0.0011501312255859375, 0.0012197494506835938, 0.0011720657348632812, 0.0011525154113769531, 0.0011610984802246094, 0.0011565685272216797, 0.0011584758758544922, 0.0011582374572753906, 0.0011565685272216797, 0.0011539459228515625, 0.001169443130493164, 0.0011556148529052734, 0.0011556148529052734, 0.001178741455078125, 0.001149892807006836, 0.001177072525024414, 0.00121307373046875, 0.0011632442474365234, 0.0011854171752929688, 0.0011639595031738281, 0.0011534690856933594, 0.0011615753173828125, 0.0011551380157470703, 0.0011546611785888672, 0.0011587142944335938, 0.001154184341430664, 0.0011620521545410156, 0.001149892807006836, 0.0011553764343261719, 0.0011625289916992188, 0.001165628433227539, 0.0011594295501708984, 0.001148223876953125, 0.0011568069458007812, 0.0011615753173828125, 0.001155853271484375, 0.0011553764343261719, 0.0011668205261230469, 0.0011570453643798828, 0.0011539459228515625, 0.0011568069458007812, 0.0011942386627197266, 0.0011897087097167969, 0.0011527538299560547, 0.0011534690856933594, 0.0012276172637939453, 0.001216888427734375, 0.0011589527130126953, 0.0011763572692871094, 0.001161813735961914, 0.0011565685272216797, 0.0011758804321289062, 0.0011610984802246094, 0.0011548995971679688, 0.0011610984802246094, 0.0011816024780273438, 0.001157999038696289, 0.0011565685272216797, 0.001153707504272461, 0.0011692047119140625, 0.0011529922485351562, 0.0011570453643798828, 0.001157522201538086, 0.0011532306671142578, 0.0011751651763916016, 0.0011506080627441406, 0.0011491775512695312, 0.0011572837829589844, 0.0011527538299560547, 0.0011518001556396484, 0.0011610984802246094, 0.0011518001556396484, 0.0011601448059082031, 0.0011684894561767578, 0.0011701583862304688, 0.001161336898803711, 0.0011568069458007812, 0.0011551380157470703, 0.0011563301086425781, 0.0011527538299560547, 0.0011508464813232422, 0.0011606216430664062, 0.0011496543884277344, 0.0011539459228515625, 0.0011539459228515625, 0.0011510848999023438, 0.0011818408966064453, 0.0011589527130126953, 0.0011563301086425781, 0.0011518001556396484, 0.0012481212615966797, 0.0011620521545410156, 0.0011565685272216797, 0.0011508464813232422, 0.0011584758758544922, 0.001153707504272461, 0.0011587142944335938, 0.0011641979217529297, 0.0012073516845703125, 0.001157522201538086, 0.0011658668518066406, 0.0011501312255859375, 0.0011491775512695312, 0.0011508464813232422, 0.001153707504272461, 0.0011687278747558594, 0.0012652873992919922, 0.001161813735961914, 0.0011718273162841797, 0.0011594295501708984, 0.0011658668518066406, 0.0011513233184814453, 0.0011510848999023438, 0.0011701583862304688, 0.0012192726135253906, 0.0012125968933105469, 0.001165151596069336, 0.0011627674102783203, 0.0011718273162841797, 0.0011620521545410156, 0.0011513233184814453, 0.001168966293334961, 0.0011568069458007812, 0.0011563301086425781, 0.001165151596069336, 0.0011472702026367188, 0.0011584758758544922, 0.0011632442474365234, 0.0011546611785888672, 0.0011551380157470703, 0.0011730194091796875, 0.0011529922485351562, 0.0011601448059082031, 0.0011484622955322266, 0.001161336898803711, 0.0011701583862304688, 0.0011510848999023438, 0.0011548995971679688, 0.001169443130493164, 0.0011794567108154297, 0.0011680126190185547, 0.0011494159698486328, 0.0011513233184814453, 0.0011608600616455078, 0.0011572837829589844, 0.0011532306671142578, 0.001153707504272461, 0.001154184341430664, 0.001177072525024414, 0.0011556148529052734, 0.001153707504272461, 0.0011639595031738281, 0.0011577606201171875, 0.001168966293334961, 0.00115966796875, 0.001154184341430664, 0.0011565685272216797, 0.0011565685272216797, 0.0011527538299560547, 0.0011610984802246094, 0.001155853271484375, 0.0011866092681884766, 0.0011646747589111328, 0.001171112060546875, 0.0011684894561767578, 0.0011489391326904297, 0.001157999038696289, 0.0011620521545410156, 0.0011560916900634766, 0.001165151596069336, 0.0011610984802246094, 0.0011532306671142578, 0.0011641979217529297, 0.0011548995971679688, 0.0011494159698486328, 0.0011963844299316406, 0.0011582374572753906, 0.0011565685272216797, 0.0011565685272216797, 0.0013074874877929688, 0.0011725425720214844, 0.0011599063873291016, 0.0011544227600097656, 0.0011639595031738281, 0.001163482666015625, 0.001154184341430664, 0.001153707504272461, 0.0011563301086425781, 0.0011749267578125, 0.0011548995971679688, 0.0011527538299560547, 0.0011622905731201172, 0.0011556148529052734, 0.0011646747589111328, 0.001171112060546875, 0.0013301372528076172, 0.0011751651763916016, 0.0011675357818603516, 0.0011568069458007812, 0.0011632442474365234, 0.0011529922485351562, 0.0011587142944335938, 0.0011668205261230469, 0.0011501312255859375, 0.0011639595031738281, 0.0013973712921142578, 0.0011661052703857422, 0.0011796951293945312, 0.0012781620025634766, 0.0011653900146484375, 0.001172780990600586, 0.0011570453643798828, 0.0011568069458007812, 0.0011761188507080078, 0.0011723041534423828, 0.0011749267578125]
print(np.max(arr))