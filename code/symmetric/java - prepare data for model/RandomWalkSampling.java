package dataPrepare.ProxEmbed;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;


/**
 * Generate samplings by random walk samplings.
 * 
 * Procedure:
 * 1.Read the whole graph
 * 2.Generate samplings by random walk.
 */
public class RandomWalkSampling {

	/**
	 * Random number generator
	 */
	private Random random=new Random(123);
	
	static String nodesPath=Config.NODES_PATH;
	static String edgesPath=Config.EDGES_PATH;
	static String savePath=Config.SAVE_PATH_FOR_RANDOMWALK_SAMPLINGS;
	static int K=Config.SAMPLING_TIMES_PER_NODE;
	static int L=Config.SAMPLING_LENGTH_PER_PATH;
	static String typeAndTypeIdPath=Config.TYPE_TYPEID_SAVEFILE;
	static int shortest_path_length=Config.SHORTEST_LENGTH_FOR_SAMPLING;
	
	public static void main(String[] args) {
		ReadWholeGraph rwg=new ReadWholeGraph();
		//1.Read the whole graph
		Map<Integer,Node> data=rwg.readDataFromFile(
				nodesPath, 
				edgesPath, 
				typeAndTypeIdPath); 
		//2.Generate samplings by random walk.
		RandomWalkSampling crws=new RandomWalkSampling();
		crws.randomWalkSampling(data, K, L, savePath);
	}

	/**
	 * Generate samplings by random walk.
	 * @param data
	 * @param k
	 * @param l
	 * @param pathsFile
	 */
	public void randomWalkSampling(Map<Integer,Node> data,int k,int l,String pathsFile){
		List<Node> path=null;
		FileWriter writer=null;
		StringBuilder sb=new StringBuilder();
		try {
			writer=new FileWriter(pathsFile);
		} catch (IOException e) {
			e.printStackTrace();
		}
		for(Node node:data.values()){
			for(int i=0;i<k;i++){
				path=randomWalkPath(node,l,data);
				if(path.size()<shortest_path_length){
					continue;
				}
				sb.delete( 0, sb.length() );
				for(int j=0;j<path.size();j++){
					sb.append(path.get(j).getId()+" ");
				}
				sb.append("\r\n");
				try {
					writer.write(sb.toString());
					writer.flush();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
			
		}
	}
	
	/**
	 * Generate a path by random walk.
	 * @param start
	 * @param l
	 * @param data
	 * @return
	 */
	private List<Node> randomWalkPath(Node start,int l, Map<Integer,Node> data){
		List<Node> path=new ArrayList<Node>(l+1);
		path.add(start);
		Node now=start;
		Set<Integer> types_set=new HashSet<Integer>();
		List<Integer> types=new ArrayList<Integer>();
		Map<Integer,List<Integer>> neighbours=new HashMap<Integer, List<Integer>>();
		int type=-1;
		List<Integer> list=null;
		for(int i=0;i<l;i++){
			if(now.out_nodes.size()==0){
				break;
			}
			types_set.clear();
			types.clear();
			neighbours.clear();
			for(Node n:now.out_nodes){
				types_set.add(n.getTypeId());
				if(neighbours.containsKey(n.getTypeId())){
					neighbours.get(n.getTypeId()).add(n.getId());
				}
				else{
					List<Integer> ids=new ArrayList<Integer>();
					ids.add(n.getId());
					neighbours.put(n.getTypeId(), ids);
				}
			}
			types.addAll(types_set);
			type=types.get(random.nextInt(types.size()));
			list=neighbours.get(type);
			now=data.get(list.get(random.nextInt(list.size())));
			path.add(now);
		}
		return path;
	}
}
