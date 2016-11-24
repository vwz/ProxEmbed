package dataPrepare.ProxEmbed;

/**
 * Prepare data from graph for training and testing.
 * 
 * Procedure:
 * 1.Sampling by random walk.
 * 2.Generate entity features by information from graph.
 * 3.Generate sub-paths from samplings from step 1.
 * 
 * Finally, these results are written to a file.
 * 
 * Attention:
 * 	If you want to generate data for symmetric training data, then you should remove "GenerateEntitiesFeaturesByGraph.main(null);" and only use "GenerateEntitiesFeatureByTypes.main(null);".
 *  While if you want to generate data for asymmetric training data, then you should remove "GenerateEntitiesFeatureByTypes.main(null);" and only use "GenerateEntitiesFeaturesByGraph.main(null);".
 */
public class Main {

	public static void main(String[] args) {
		//random walk sampling
		long starttime=System.currentTimeMillis();
		System.out.println("Start random walk sampling......");
		RandomWalkSampling.main(null);
		
		//generate entities features by information from this graph.
		System.out.println("Start generating entities' features......");
//		GenerateEntitiesFeaturesByGraph.main(null);//Generate entity features by information from neighbours -- just for asymmetric
		GenerateEntitiesFeatureByTypes.main(null);//Generate entity features only by type information -- just for symmetric
		
		//generate sub-paths
		System.out.println("Start generating sub-paths from samplings......");
		GenerateSubPathsFromSamplings.main(null);//Generate sub-paths from samplings.
		long endtime=System.currentTimeMillis();
		System.out.println("Cost time == "+(endtime-starttime)+" ms");
	}

}
