/**
 * Created by Nick on 6/26/2018
 */

function getCollatzNetwork(iterationCount) {
    var nodes = [];
    var newNodes = [];
    var edges = [];

    // add 1 up front, have loop just look at new nodes
    newNodes.push({
        id: 1,
        label: String(1)
    });

    for (var i = 0; i < iterationCount; i++) {
        var theseNodes = newNodes;
        newNodes = [];

        for (var j = 0; j < theseNodes.length; j++) {
            nodes.push(theseNodes[j]);
            var value = theseNodes[j].id;
            
            // always can multiply by 2
            valueEven = 2*value;
            newNodes.push({
                id: valueEven,
                label: String(valueEven)
            });

            edges.push({
                from: value,
                to: valueEven
            });

            // check if we can subtract 1 and divide by 3
            if (value % 2 == 0
                && (value - 1) % 3 == 0
                && (value - 1) / 3 > 1) {
                // yes, we can
                valueOdd = (value - 1) / 3;
                newNodes.push({
                    id: valueOdd,
                    label: String(valueOdd)
                });

                edges.push({
                    from: value,
                    to: valueOdd
                });
            }
        }
    }

    return {nodes:nodes, edges:edges};
}

