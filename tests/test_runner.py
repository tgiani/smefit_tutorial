# -*- coding: utf-8 -*-
import pathlib
import shutil

import smefit_tutorial.runner

runcard_folder = commondata_path = pathlib.Path(__file__).parents[0]


class TestRunner:

    test_runner = smefit_tutorial.runner.Runner.from_file(runcard_folder / "test_runcard.yaml")

    def test_init(self):
        assert self.test_runner.run_card["datasets"][0] == "test_dataset"
        assert self.test_runner.run_card["result_ID"] == "test_runcard"

    def test_setup_result_folder(self):
        result_folder = pathlib.Path(self.test_runner.run_card["result_path"])
        res_folder_fit = result_folder / self.test_runner.run_card["result_ID"]
        assert res_folder_fit.exists()
        shutil.rmtree(res_folder_fit)
