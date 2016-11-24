package dataPrepare.ProxEmbed;

import java.io.FileWriter;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/**
 * Generate entity features by information from neighbours -- just for asymmetric
 */
public class GenerateEntitiesFeaturesByGraph {

	private Set<String> types=new HashSet<String>();
	private Map<String,Integer> type2Typeid=new HashMap<String, Integer>();
	private Map<Integer,String> typeid2Type=new HashMap<Integer, String>();
	static String nodes_path=Config.NODES_PATH;
	static String edges_path=Config.EDGES_PATH;
	static String entities_feature_file=Config.NODES_FEATURE_SAVE_PATH;
	static String typeAndTypeIdPath=Config.TYPE_TYPEID_SAVEFILE;
	static double feature_type_value=Config.FEATURE_TYPE_VALUE;
	
	public static void main(String[] args) {
		ReadWholeGraph rwg=new ReadWholeGraph();
		Map<Integer,Node> graph=rwg.readDataFromFile(nodes_path, edges_path, typeAndTypeIdPath);
		GenerateEntitiesFeaturesByGraph gefb=new GenerateEntitiesFeaturesByGraph();
		gefb.analyseTypes(graph);
		gefb.generateFeaturesByGraph(graph, entities_feature_file,feature_type_value);
	}

	/**
	 * Analyse nodes types
	 */
	public void analyseTypes(Map<Integer,Node> graph){
		for(Node n:graph.values()){
			types.add(n.getType());
			if(!type2Typeid.containsKey(n.getType())){
				type2Typeid.put(n.getType(), type2Typeid.size());
				typeid2Type.put(typeid2Type.size(), n.getType());
			}	
		}
	}
	
	/**
	 * Generate nodes features
	 */
	public void generateFeaturesByGraph(Map<Integer,Node> graph,String saveFile,double typeValue){
		int dimension=types.size()+1+types.size()+1;
		int nodesNum=graph.size();
		StringBuilder sb=new StringBuilder();
		String type=null;
		int typeId=0;
		double value=0;
		double sum=0;
		Map<String,Integer> typesNum=new HashMap<String, Integer>();
		FileWriter writer = null;
		try {
			writer = new FileWriter(saveFile);
			writer.write(nodesNum+" "+dimension+"\r\n");
			writer.flush();
			for(Node now:graph.values()){
				sb.delete( 0, sb.length() );
				typesNum.clear();
				
				sb.append(now.getId()+" ");
				type=now.getType();
				typeId=type2Typeid.get(type);
				
				for(int i=0;i<types.size();i++){
					if(i==typeId){
						sb.append(typeValue+" ");
					}
					else{
						sb.append(0.0+" ");
					}
				}
				
				value=now.in_nodes.size();
				sb.append(Math.log(value+1.0)+" ");
				
				for(Node n:now.in_nodes){
					type=n.getType();
					if(typesNum.containsKey(type)){
						typesNum.put(type, typesNum.get(type)+1);
					}
					else{
						typesNum.put(type, 1);
					}
				}
				for(int i=0;i<typeid2Type.size();i++){
					type=typeid2Type.get(i);
					if(typesNum.containsKey(type)){
						sb.append(Math.log(typesNum.get(type)+1)+" ");
					}
					else{
						sb.append(0.0+" ");
					}
				}
				
				value=0;
				sum=0;
				for(int num:typesNum.values()){
					value=(num+0.0)/now.in_nodes.size();
					sum+=-value*Math.log(value);
				}
				sb.append(sum);
				
				sb.append("\r\n");
				writer.write(sb.toString());
				writer.flush();
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		finally{
			try {
				if(writer!=null){
					writer.close();
					writer=null;
				}
			} catch (Exception e2) {
				e2.printStackTrace();
			}
		}
	}
}
