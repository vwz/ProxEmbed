package dataPrepare.ProxEmbed;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.Map;

/**
 * Read the while graph, and then save the info into Map<Integer,Node>
 */
public class ReadWholeGraph {

	static Map<Integer,String> typeid2Type=new HashMap<Integer, String>();
	static Map<String,Integer> type2Typeid=new HashMap<String, Integer>();

	/**
	 * Read whole graph info
	 * @param nodesPath
	 * @param edgesPath
	 * @param typeAndTypeIdPath
	 * @return
	 */
	public Map<Integer,Node> readDataFromFile(String nodesPath,String edgesPath,String typeAndTypeIdPath){
		Map<Integer,Node> data=new HashMap<Integer,Node>();
		BufferedReader br=null;
		String[] arr=null;
		Node node=null;
		try {
			br = new BufferedReader(new InputStreamReader(new FileInputStream(nodesPath), "UTF-8"));
			String temp = null;
			while ((temp = br.readLine()) != null ) {
				temp=temp.trim();
				if(temp.length()>0){
					arr=temp.split("\t");
					node=new Node();
					node.setId(Integer.parseInt(arr[0]));
					node.setType(arr[1]);
					if(type2Typeid.containsKey(arr[1])){
						node.setTypeId(type2Typeid.get(arr[1]));
					}
					else{
						type2Typeid.put(arr[1], type2Typeid.size());
						typeid2Type.put(typeid2Type.size(), arr[1]);
						node.setTypeId(type2Typeid.get(arr[1]));
					}
					data.put(Integer.parseInt(arr[0]), node);
				}
			}
		} catch (Exception e2) {
			e2.printStackTrace();
		}
		finally{
			try {
				if(br!=null){
					br.close();
					br=null;
				}
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		int start=0;
		int end=0;
		Node startNode=null;
		Node endNode=null;
		try {
			br = new BufferedReader(new InputStreamReader(new FileInputStream(edgesPath), "UTF-8"));
			String temp = null;
			while ((temp = br.readLine()) != null ) {
				temp=temp.trim();
				if(temp.length()>0){
					arr=temp.split("\t");
					start=Integer.parseInt(arr[0]);
					end=Integer.parseInt(arr[1]);
					startNode=data.get(start);
					endNode=data.get(end);
					startNode.out_ids.add(end);
					startNode.out_nodes.add(endNode);
					endNode.in_ids.add(start);
					endNode.in_nodes.add(startNode);
				}
			}
		} catch (Exception e2) {
			e2.printStackTrace();
		}
		finally{
			try {
				if(br!=null){
					br.close();
					br=null;
				}
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		FileWriter writer = null;
		try {
			writer = new FileWriter(typeAndTypeIdPath);
			for(String type:type2Typeid.keySet()){
				writer.write(type+" "+type2Typeid.get(type)+"\r\n");
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
		
		return data;
	}
}
