# name=Mixing-AI Super Bridge
# Version: 7.2 (Dynamic Targeting)
import mixer
import plugins
import device
import ui

# Глобальный кеш для таргетинга
target_slot = -1

def OnInit():
    print("Mixing-AI Bridge v7.2: Dynamic Targeting Online")

def OnSysEx(event):
    global target_slot
    # Заголовок: F0 00 4D 41 49
    data = event.sysex
    if len(data) > 5 and data[1] == 0x00 and data[2] == 0x4D and data[3] == 0x41 and data[4] == 0x49:
        try:
            msg = "".join([chr(b) for b in data[5:-1]])
            print("SysEx Command: " + msg)
            
            if msg == "SELECT_EMPTY":
                found = False
                for i in range(1, 101):
                    empty = True
                    for slot in range(10):
                        if mixer.isTrackPluginValid(i, slot):
                            empty = False
                            break
                    if empty:
                        mixer.setTrackNumber(i)
                        ui.showWindow(2)
                        print("Action: Selected empty track " + str(i))
                        found = True
                        break
                if not found:
                    print("Error: No empty tracks found!")
            
            elif msg == "OPEN_PICKER":
                ui.showWindow(5)
                print("Action: Opened Plugin Picker")

            elif msg.startswith("TARGET_PLUGIN|"):
                name_to_find = msg.split("|")[1]
                track = mixer.trackNumber()
                target_slot = -1 # Сброс
                for s in range(10):
                    p_name = plugins.getPluginName(track, s)
                    if name_to_find.lower() in p_name.lower():
                        target_slot = s
                        print("Action: Targeted '%s' at Slot %d" % (p_name, s + 1))
                        break
                if target_slot == -1:
                    print("Error: Plugin '%s' not found on track %d" % (name_to_find, track))

            event.handled = True
        except Exception as e:
            print("SysEx Error: " + str(e))

def OnMidiMsg(event):
    global target_slot
    if (event.status & 0xF0) == 0xB0:
        # Если канал MIDI 16 (0x0F), используем закешированный target_slot
        # Иначе используем канал как индекс слота (для обратной совместимости)
        midi_chan = event.status & 0x0F
        
        slot_to_use = target_slot if midi_chan == 0x0F else midi_chan
        
        if slot_to_use == -1:
            # Если таргет не задан, пробуем найти EQ по умолчанию
            track = mixer.trackNumber()
            for s in range(10):
                if "eq" in plugins.getPluginName(track, s).lower():
                    slot_to_use = s
                    break

        param_index = event.data1
        param_value = event.data2 / 127.0
        track_index = mixer.trackNumber()
        
        if slot_to_use != -1 and mixer.isTrackPluginValid(track_index, slot_to_use):
            plugins.setParamValue(param_value, param_index, track_index, slot_to_use)
            # print("MIDI Param: Track %d, Slot %d, Param %d, Val %.2f" % (track_index, slot_to_use + 1, param_index, param_value))
            event.handled = True
