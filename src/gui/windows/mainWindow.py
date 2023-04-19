from ..guiUtils import *
from ..descriptions import DESC
from ...utils import *
from ...Classes import prngs
from ...Classes.Runner import Runner
from ...Classes import parsers


class mainWindow:

    def __init__(self):
        char_width = 7  # default font size
        window_width = 133 * char_width
        combo_width = 25
        txt_width = combo_width + 2

        leftCol = psg.Column([
            horizontalCenter([psg.Text("PRNG :", size=txt_width), psg.Text("Test type :", size=txt_width)]),
            horizontalCenter([
                psg.Combo([], readonly=True, key="-prngselection-", size=combo_width, enable_events=True),
                psg.Combo([], readonly=True, key="-typeelection-", size=combo_width, enable_events=True)]),
            [psg.Text("")],
            horizontalCenter([
                toggleableFileBrowse("Select file", target="-path-", enable_events=True, size=10, key="-fileselect-"),
                psg.Text("", size=(txt_width + 14, 2), key="-path-")]),
            [psg.Text("")],
            horizontalCenter([psg.Text("Format :", size=txt_width, key="-formatlabel-"), psg.Text("", size=txt_width)]),
            horizontalCenter([
                toggleableCombo([], readonly=True, key="-formatselection-", size=combo_width, enable_events=True),
                psg.Text("", size=txt_width)]), [psg.Text("")],
            [psg.Text("", pad=(0, 0))],  # pad makes it exactly aligned

        ])

        rightCol = psg.Column([
            horizontalCenter([
                psg.Checkbox(" Run individual test", size=combo_width, enable_events=True, key="-checkbox-"),
                psg.Text("Test to run :", size=txt_width, key="-testlabel-")]),
            horizontalCenter([
                psg.Text("", size=txt_width),
                toggleableCombo([], readonly=True, key="-testselection-", size=combo_width, enable_events=True)]),
            horizontalCenter([psg.Text("Description :", size=56)]),
            horizontalCenter([psg.Multiline("", size=(56, 10), disabled=True, no_scrollbar=True, key="-desc-",
                                            border_width=0)]),

        ])

        layout = [
            [psg.VPush()],
            [leftCol, rightCol],
            # end cpsg.VPush()olumn
            [psg.Push(), psg.Button("Run", button_color="green3"), psg.Button("Clear logs"), psg.Push()],
            horizontalCenter([
                psg.pin(psg.ProgressBar(0, size_px=(125 * char_width, 3), key="-pbar-", bar_color=("green3", "")))]),
            horizontalCenter([psg.pin(psg.ProgressBar(0, size_px=(125 * char_width, 3), key="-pbar2-"))]),
            horizontalCenter([psg.Text("Logs :", size=(125, 0))]),
            horizontalCenter([psg.Multiline(size=(125, 20), reroute_cprint=True, reroute_stdout=True, disabled=True,
                                            no_scrollbar=True, autoscroll=True, key="-log-")]),
            [psg.VPush()],
        ]

        self.window = psg.Window(title="RNG Validator", layout=layout, size=(window_width, 650), finalize=True)

        # initial state
        self.selectedType = self.populateType()
        self.selectedPrng = self.populatePRNGS()
        self.selectedFormat = self.prngChanged(self.selectedPrng)
        self.selectedTest = self.testTypeChanged(self.selectedType)

        self.window["-desc-"].update(background_color=psg.theme_background_color())
        self.window["-desc-"].update(value="Click on an element to show a description.")
        self.window["-pbar-"].update(current_count=0, visible=False)
        self.window["-pbar2-"].update(current_count=0, visible=False)

    def setDescription(self, kind, identifier):
        txt = "No description available."
        if DESC.get(kind) is not None:
            if DESC[kind].get(identifier) is not None:
                txt = DESC[kind][identifier]
        self.window["-desc-"].update(value=txt)

    def parserChanged(self, itemName):
        """
        This function is called when the file format selection has changed.

        :param itemName: The newly selected file format's name.
        :type itemName: str
        """
        chosenParser = strToEnum(PARSERS, itemName)
        self.setDescription("Parser", chosenParser)

    def populateType(self):
        """
        Populates the test type DropDown widget with all supported test types.
        You shouldn't have to change this.
        """
        select = self.window["-typeelection-"]
        values = []
        for t in TEST_TYPES:
            values.append(t.name)
        select.update(values=values, value=values[0])
        return values[0]

    def populatePRNGS(self):
        """
        Populates the PRNG DropDown widget with all supported PRNGs.
        You shouldn't have to change this.
        """
        select = self.window["-prngselection-"]
        values = []
        for t in PRNGS:
            values.append(prngs.PRNG_DISPLAY_NAME[t])
        select.update(values=values, value=values[0])
        return values[0]

    def prngChanged(self, itemName):
        """
        This function is called when the PRNG selection has changed.

        :param itemName: The newly selected test type's name.
        :type itemName: str
        """

        chosenPRNG = prngs.PRNG_NAME_TO_ENUM[itemName]
        self.setDescription("PRNG", chosenPRNG)
        select = self.window["-formatselection-"]
        values = []

        if chosenPRNG == PRNGS.External:
            for t in PARSERS:
                values.append(t.name)

            select.update(values=values, value=values[0])
            self.window["-formatlabel-"].update(value="Format :")
            self.window["-formatselection-"].update(visible=True)
            self.window["-fileselect-"].update(visible=True)
            return values[0]

        else:
            select.update(values=values)
            self.window["-formatlabel-"].update(value="")
            self.window["-formatselection-"].update(visible=False)
            self.window["-fileselect-"].update(visible=False)
            self.window["-path-"].update(value="")
            return None

    def testTypeChanged(self, itemName, updateDesc=True):
        """
        This function is called when the test type selection has changed.
        Depending on the test type, different individual tests have to be displayed.

        :param itemName: The newly selected test type's name.
        :type itemName: str
        """

        chosenType = strToEnum(TEST_TYPES, itemName)
        values = []

        if updateDesc:
            self.setDescription("Type", chosenType)

        if not self.window["-checkbox-"].get():
            self.window["-testselection-"].update(visible=False)
            self.window["-testlabel-"].update(value="")
            return None

        if chosenType == TEST_TYPES.BSI:
            for t in BSI_TESTS:
                values.append(t.name)
            self.window["-testselection-"].update(visible=True, values=values, value=values[0])
            self.window["-testlabel-"].update(value="Test to run :")
            return values[0]

        if chosenType == TEST_TYPES.NIST:
            for t in NIST_TESTS:
                values.append(t.name)
            self.window["-testselection-"].update(visible=True, values=values, value=values[0])
            self.window["-testlabel-"].update(value="Test to run :")
            return values[0]

        else:
            self.window["-testselection-"].update(visible=False)
            self.window["-testlabel-"].update(value="")
            return None

    def individualCheckChanged(self, checked, selectedType):
        self.setDescription("Check", checked)
        return self.testTypeChanged(selectedType, updateDesc=False)

    def testChanged(self, selectedType, selectedTest):
        """
        This function is called when the individual test selection has changed.
        """
        chosenTestType = strToEnum(TEST_TYPES, selectedType)
        if chosenTestType == TEST_TYPES.BSI:
            chosenTest = strToEnum(BSI_TESTS, selectedTest)
        elif chosenTestType == TEST_TYPES.NIST:
            chosenTest = strToEnum(NIST_TESTS, selectedTest)
        else:
            raise NotImplementedError("Unknown test type : {}".format(selectedType))
        self.setDescription("Test", chosenTest)
        return selectedTest

    def resetInternalProgressBar(self, total):
        pbar = self.window["-pbar2-"]
        self.pbar2_count = 0
        self.pbar2_updateCount = 0
        self.pbar2_total = total
        pbar.update(max=total, current_count=self.pbar2_count)
        self.window.refresh()

    def incrementInternalProgressBar(self):
        self.pbar2_count += 1
        self.pbar2_updateCount += 1
        # update every 20th of total count
        # updating is very costly in perf
        if self.pbar2_updateCount >= 1:  # pbar2_total // 20:
            self.pbar2_updateCount = 0
            pbar = self.window["-pbar2-"]
            pbar.update(current_count=self.pbar2_count)
            self.window.refresh()

    def run(self, path):
        """
        This function is called when the user clicks on the "Run" button.
        """

        # there is always a PRNG
        chosenPRNG = prngs.PRNG_NAME_TO_ENUM[self.selectedPrng]

        if chosenPRNG == PRNGS.External:
            # Get the file path
            if path == "":
                perror("You must choose a file containing the PRNG's output data.")
                return None
            print("Running tests against an external PRNG by reading random data from : {}".format(path))
            # get the parser
            parser = strToEnum(PARSERS, self.selectedFormat)
            parser = parsers.PARSER_CLASS[parser]
            # Instanciate the PRNG
            p = prngs.PRNG_CLASS[chosenPRNG](path, parser)
            # Check that the parsing succeeded
            if p.pool is None:
                perror("Something wen't wrong during parsing of the file : {}".format(path))
                return None
        else:
            print("Running tests against {}".format(prngs.PRNG_DISPLAY_NAME[chosenPRNG]))
            p = prngs.PRNG_CLASS[chosenPRNG]()
        # Instanciate the runner
        r = Runner(p)

        # there is always a test type
        chosenTestType = strToEnum(TEST_TYPES, self.selectedType)

        if self.selectedTest is not None:
            # Run a single test
            if chosenTestType == TEST_TYPES.BSI:
                # BSI
                chosenTest = strToEnum(BSI_TESTS, self.selectedTest)
                r.singleBSITest(chosenTest)
            elif chosenTestType == TEST_TYPES.NIST:
                # NIST
                chosenTest = strToEnum(NIST_TESTS, self.selectedTest)
                r.singleNISTTest(chosenTest)
            else:
                pwarning("Unknown test type : {}".format(chosenTestType.name))
                return None
        else:
            # Run the procedures
            if chosenTestType == TEST_TYPES.BSI:
                # BSI
                r.runBSITests(self)
            elif chosenTestType == TEST_TYPES.NIST:
                # NIST
                r.runNISTTests(self)
            else:
                pwarning("Unknown test type : {}".format(chosenTestType.name))
                return None

    def handleEvents(self, event, values):
        if event == "-formatselection-":
            self.selectedFormat = values["-formatselection-"]
            self.parserChanged(self.selectedFormat)

        if event == "-prngselection-":
            self.selectedPrng = values["-prngselection-"]
            self.selectedFormat = self.prngChanged(self.selectedPrng)

        if event == "-typeelection-":
            self.selectedType = values["-typeelection-"]
            self.selectedTest = self.testTypeChanged(self.selectedType)

        if event == "-testselection-":
            self.selectedTest = self.testChanged(self.selectedType, values["-testselection-"])

        if event == "-checkbox-":
            checked = values["-checkbox-"]
            self.selectedTest = self.individualCheckChanged(checked, self.selectedType)

        if event == "Run":
            path = self.window["-path-"].get()
            self.run(path)

        if event == "Clear logs":
            self.window["-log-"].update("")
