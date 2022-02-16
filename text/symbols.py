""" from https://github.com/keithito/tacotron """

'''
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run
through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details.

For Chinese Mandarin, the set of symbols can be switched to Pinyin initials, finals, retroflex (Erhua), 
tones and prosodic structure tags.

'''

from . import cmudict
from . import pinyin

_pad        = '_'
_eos        = '~'
_pts = ',.:;?! %-/\'\"()'
_characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_digits = '012346789'

# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
_puncts = ['@' + s for s in _pts]
_arpabet = ['@' + s for s in cmudict.valid_symbols]
_pinyin = ["@" + s for s in pinyin.valid_symbols]

symbols = [_pad, _eos] + list(_puncts) + list(_characters) + list(_digits) + _arpabet + _pinyin

