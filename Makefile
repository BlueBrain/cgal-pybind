
clean-pyc:
	find cgal_pybind -type f -name "*.py[co]" -o -name __pychache__ -exec rm -rf {} +

clean-cpp:
	find cgal_pybind -type f -name '*.c' -delete -o -name '*.o' -delete -o -name '*.so' -delete -o -name '*.cpp' -delete

clean-general:
	find cgal_pybind -type f -name .DS_Store -delete -o -name .idea -delete -o -name '*~' -delete

clean-build:
	rm -rf build dist *.egg-info


.PHONY: clean
clean: clean-build clean-general clean-cpp clean-pyc

.PHONY: install
install: clean
	pip3 install setuptools pip --upgrade numpy
	pip3 install -e .
