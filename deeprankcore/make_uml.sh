#!/bin/bash
###############################################################################
# To generate UML class diagrams (.svg) using PlantUML
#
# Requirements:
#  1. pylint: https://pypi.org/project/pylint/
#  2. plantuml: https://formulae.brew.sh/formula/plantuml
#
# Notice:
#  - Run the script in the `deeprank-core/deeprankcore`
#  - Do keep .svg files in the same folder to view diagrams interactively
###############################################################################

# Create output dir
[[ ! -d uml ]] && mkdir uml
[[ -d uml ]] && rm -f uml/*
cd uml

# Generate .puml for the project
# output: packages_npl.pul and classes_npl.puml
echo "Generating PlantUML .puml files..."
pyreverse -my -o puml --ignore tests -p npl ..

# Generate .puml for all classes
# output: {classname}.puml
eval "$(awk '
    BEGIN{print "pyreverse -a 2 -s 1 -o puml --ignore tests -p npl \\"}
    /^class/{print " -c", $4, "\\"}
    END{print ".."}' classes_npl.puml)"

# Add .svg link to .puml files
# this will enable interactive viewing of UML
for i in `awk '/^class/{print $4}END{print "classes_npl"}' classes_npl.puml`; do
    awk '/^class \"deeprankcore/ { print $1, $2, $3, $4, "[[" $4 ".svg]]", $5}
        !/^class \"deeprankcore/ {print}' $i.puml > $i.link.puml
    mv $i.link.puml $i.puml
done

# Generate UML .svg images
# Output: {classname}.svg, UML for each class
printf "\nGenerating UML .svg diagrams:\n"
for i in `awk '/^class/{print $4}' classes_npl.puml`; do
    cat $i.puml | plantuml -tsvg -pipe > $i.svg
    echo "  uml/$i.svg"
done

# Output:
#  packages_npl.svg, UML package diagram (modules)
#  classes_npl.svg, UML for all classes
plantuml -tsvg -I classes_npl.puml -I packages_npl.puml
echo "  uml/classes_npl.svg"
echo "  uml/packages_npl.svg"
echo "Done"