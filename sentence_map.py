# sentence_map.py

ISL_TO_HI = {

    # ---------------- BASIC HELP ----------------
    frozenset(["help"]): "मदद चाहिए",
    frozenset(["self", "help"]): "मुझे मदद चाहिए",
    frozenset(["you", "help"]): "मदद चाहिए",

    frozenset(["please", "help"]): "कृपया मदद करें",
    frozenset(["please", "self", "help"]): "कृपया मेरी मदद करें",

    # ---------------- LOCATION ----------------
    frozenset(["help", "here"]): "यहाँ मदद चाहिए",
    frozenset(["help", "there"]): "वहाँ मदद चाहिए",
    frozenset(["self", "here", "help"]): "मैं यहाँ हूँ, मदद चाहिए",
    frozenset(["self", "there", "help"]): "मैं वहाँ हूँ, मदद चाहिए",

    # ---------------- DOCTOR ----------------
    frozenset(["doctor"]): "डॉक्टर चाहिए",
    frozenset(["help", "doctor"]): "मुझे डॉक्टर की मदद चाहिए",
    frozenset(["self", "doctor"]): "मुझे डॉक्टर चाहिए",
    frozenset(["self", "help", "doctor"]): "मुझे डॉक्टर की मदद चाहिए",

    frozenset(["doctor", "here"]): "यहाँ डॉक्टर चाहिए",
    frozenset(["doctor", "there"]): "वहाँ डॉक्टर को बुलाइए",

    # ---------------- PAIN ----------------
    frozenset(["pain"]): "मुझे दर्द है",
    frozenset(["self", "pain"]): "मुझे दर्द है",
    frozenset(["pain", "help"]): "मुझे दर्द है, मदद करें",
    frozenset(["pain", "doctor"]): "मुझे दर्द है, डॉक्टर चाहिए",

    frozenset(["pain", "here"]): "यहाँ दर्द है",
    frozenset(["pain", "there"]): "वहाँ दर्द है",

    # ---------------- ACCIDENT ----------------
    frozenset(["accident"]): "दुर्घटना हुई है",
    frozenset(["accident", "help"]): "दुर्घटना हुई है, मदद चाहिए",
    frozenset(["accident", "doctor"]): "दुर्घटना हुई है, डॉक्टर को बुलाइए",

    frozenset(["accident", "pain"]): "दुर्घटना हुई है, मुझे दर्द है",
    frozenset(["accident", "pain", "help"]):
        "दुर्घटना हुई है, मुझे दर्द है, मदद करें",

    frozenset(["accident", "pain", "doctor"]):
        "दुर्घटना हुई है, मुझे दर्द है, डॉक्टर को बुलाइए",

    frozenset(["accident", "here"]):
        "यहाँ दुर्घटना हुई है",

    frozenset(["accident", "there"]):
        "वहाँ दुर्घटना हुई है",

    # ---------------- CALL ----------------
    frozenset(["call", "help"]): "मदद के लिए कॉल करें",
    frozenset(["call", "doctor"]): "डॉक्टर को कॉल करें",
    frozenset(["call", "help", "doctor"]): "डॉक्टर को कॉल करके मदद करें",

    # ---------------- THIEF ----------------
    frozenset(["thief"]): "चोर है",
    frozenset(["thief", "help"]): "चोर है, मदद करें",
    frozenset(["thief", "stop"]): "चोर को रोकिए",

    frozenset(["thief", "here"]): "यहाँ चोर है",
    frozenset(["thief", "there"]): "वहाँ चोर है",

    frozenset(["thief", "here", "help"]):
        "यहाँ चोर है, मदद करें",

    frozenset(["thief", "stop", "help"]):
        "चोर को रोकिए, मदद करें",

    frozenset(["thief", "call"]):
        "चोर है, कॉल करें",

    frozenset(["thief", "call", "police"]):
        "चोर है, पुलिस को कॉल करें",

}

