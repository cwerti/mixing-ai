# Mixing-AI: Auto-EQ Setup Script
import plugins
import mixer

def setup_eq():
    # Индекс выбранного трека в микшере
    track = mixer.trackNumber()
    # Ищем Fruity Parametric EQ 2 в первых 10 слотах
    found = False
    for slot in range(10):
        name = plugins.getPluginName(track, slot)
        if 'Parametric EQ 2' in name:
            print(f'Found EQ 2 in slot {slot}')
            # Установка параметров
            plugins.setParamValue(0.6281, 0, slot, track)  # Band 1 Freq
            plugins.setParamValue(0.5340, 1, slot, track)  # Band 1 Gain
            plugins.setParamValue(0.7694, 5, slot, track)  # Band 2 Freq
            plugins.setParamValue(0.5308, 6, slot, track)  # Band 2 Gain
            plugins.setParamValue(0.8360, 10, slot, track)  # Band 3 Freq
            plugins.setParamValue(0.5867, 11, slot, track)  # Band 3 Gain
            plugins.setParamValue(0.8796, 15, slot, track)  # Band 4 Freq
            plugins.setParamValue(0.6174, 16, slot, track)  # Band 4 Gain
            plugins.setParamValue(0.9126, 20, slot, track)  # Band 5 Freq
            plugins.setParamValue(0.6568, 21, slot, track)  # Band 5 Gain
            plugins.setParamValue(0.9390, 25, slot, track)  # Band 6 Freq
            plugins.setParamValue(0.6745, 26, slot, track)  # Band 6 Gain
            plugins.setParamValue(0.9609, 30, slot, track)  # Band 7 Freq
            plugins.setParamValue(0.6828, 31, slot, track)  # Band 7 Gain
            found = True
            break
    if not found:
        print('Error: Fruity Parametric EQ 2 not found on selected track!')

setup_eq()